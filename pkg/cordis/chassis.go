package cordis

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"sync"
	"time"

	"github.com/huponex/hpx-body/pkg/oasis"
	"github.com/huponex/hpx-body/pkg/pips"
	"github.com/huponex/hpx-body/pkg/router"
	"github.com/huponex/hpx-body/pkg/spine"
	"github.com/huponex/hpx-body/pkg/urn"
)

type RunState string

const (
	StateIdle          RunState = "idle"
	StatePlanning      RunState = "planning"
	StateExecuting     RunState = "executing"
	StateGateRequested RunState = "gate_requested"
	StateVerifying     RunState = "verifying"
	StateCompleted     RunState = "completed"
	StateFailed        RunState = "failed"
	StateCancelled     RunState = "cancelled"
)

type ToolInvocation struct {
	BeatID         string                 `json:"beat_id"`
	ActorID        string                 `json:"actor_id"`
	OrgID          string                 `json:"org_id"`
	ToolName       string                 `json:"tool_name"`
	ExternalTarget string                 `json:"external_target,omitempty"` // Checked by ONBP EgressGuard before exec
	Params         map[string]interface{} `json:"params,omitempty"`
	Timeout        time.Duration          `json:"timeout,omitempty"`
}

type ToolResult struct {
	ToolName   string                 `json:"tool_name"`
	State      RunState               `json:"state"`
	Output     map[string]interface{} `json:"output,omitempty"`
	LatencyMS  int64                  `json:"latency_ms"`
	Recovered  bool                   `json:"recovered_from_panic,omitempty"`
	StackTrace string                 `json:"stack_trace,omitempty"`
}

type Tool interface {
	Name() string
	Description() string
	Execute(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error)
}

type ToolFunc struct {
	ToolName string
	Desc     string
	Fn       func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error)
}

func (f ToolFunc) Name() string        { return f.ToolName }
func (f ToolFunc) Description() string { return f.Desc }
func (f ToolFunc) Execute(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
	return f.Fn(ctx, params)
}

type Plugin interface {
	Name() string
	Register(c *Chassis) error
}

type ChassisConfig struct {
	AgentURN       string
	DefaultTimeout time.Duration
	Spine          *spine.Client
	Meter          *pips.Meter
	Router         *router.Router
	EgressGuard    *oasis.EgressGuard
}

type Chassis struct {
	mu             sync.RWMutex
	agentURN       string
	defaultTimeout time.Duration
	state          RunState
	spine          *spine.Client
	meter          *pips.Meter
	router         *router.Router
	guard          *oasis.EgressGuard
	tools          map[string]Tool
	plugins        []string
}

func NewChassis(cfg ChassisConfig) *Chassis {
	agentURN := urn.CanonicalOrRaw(cfg.AgentURN)
	if agentURN == "" {
		agentURN = urn.BodyURN
	}
	timeout := cfg.DefaultTimeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &Chassis{
		agentURN:       agentURN,
		defaultTimeout: timeout,
		state:          StateIdle,
		spine:          cfg.Spine,
		meter:          cfg.Meter,
		router:         cfg.Router,
		guard:          cfg.EgressGuard,
		tools:          make(map[string]Tool),
	}
}

func (c *Chassis) State() RunState {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.state
}

func (c *Chassis) Router() *router.Router      { return c.router }
func (c *Chassis) Spine() *spine.Client        { return c.spine }
func (c *Chassis) Meter() *pips.Meter          { return c.meter }
func (c *Chassis) Guard() *oasis.EgressGuard   { return c.guard }
func (c *Chassis) AgentURN() string            { return c.agentURN }

func (c *Chassis) RegisterTool(t Tool) error {
	if t == nil || t.Name() == "" {
		return errors.New("tool name is required")
	}
	c.mu.Lock()
	c.tools[t.Name()] = t
	c.mu.Unlock()
	return nil
}

func (c *Chassis) RegisterPlugin(p Plugin) error {
	if p == nil || p.Name() == "" {
		return errors.New("plugin name is required")
	}
	if err := p.Register(c); err != nil {
		return fmt.Errorf("register plugin %q: %w", p.Name(), err)
	}
	c.mu.Lock()
	c.plugins = append(c.plugins, p.Name())
	c.mu.Unlock()
	return nil
}

func (c *Chassis) transition(next RunState, inv ToolInvocation, detail string) {
	c.mu.Lock()
	prev := c.state
	c.state = next
	c.mu.Unlock()

	if c.spine != nil {
		_ = c.spine.PublishEnvelope("os.body.state", "body.state.transition", c.agentURN, inv.ActorID, inv.OrgID, map[string]interface{}{
			"from":      string(prev),
			"to":        string(next),
			"beat_id":   inv.BeatID,
			"tool_name": inv.ToolName,
			"detail":    detail,
		})
	}
}

// ExecuteTool drives the Cordis/dsh tool-call state machine with ONBP egress verification,
// context timeout/cancellation enforcement, structured panic recovery, and PIPs metering.
func (c *Chassis) ExecuteTool(ctx context.Context, inv ToolInvocation) (*ToolResult, error) {
	c.mu.RLock()
	tool, ok := c.tools[inv.ToolName]
	c.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("tool %q not registered in BODY chassis", inv.ToolName)
	}

	c.transition(StatePlanning, inv, "validating tool and ONBP egress policy")

	// 1. Native ONBP v1.0.0 Egress Guard check
	if inv.ExternalTarget != "" && c.guard != nil {
		if _, err := c.guard.CheckEgress(ctx, inv.ExternalTarget, inv.BeatID, inv.ActorID, inv.OrgID); err != nil {
			c.transition(StateGateRequested, inv, err.Error())
			if c.meter != nil {
				_ = c.meter.EmitToolCall(ctx, inv.ActorID, inv.OrgID, inv.ToolName, "blocked_unbound_threat", 0, map[string]interface{}{
					"external_target": inv.ExternalTarget,
					"beat_id":         inv.BeatID,
				})
			}
			return &ToolResult{ToolName: inv.ToolName, State: StateGateRequested}, err
		}
	}

	// 2. Enforce timeout / cancellation lifecycle
	timeout := inv.Timeout
	if timeout <= 0 {
		timeout = c.defaultTimeout
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	c.transition(StateExecuting, inv, "executing registered tool")
	start := time.Now()

	type execOutcome struct {
		out       map[string]interface{}
		err       error
		recovered bool
		stack     string
	}
	done := make(chan execOutcome, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- execOutcome{
					err:       fmt.Errorf("panic recovered in tool %q: %v", inv.ToolName, r),
					recovered: true,
					stack:     string(debug.Stack()),
				}
			}
		}()
		out, err := tool.Execute(runCtx, inv.Params)
		done <- execOutcome{out: out, err: err}
	}()

	var res execOutcome
	select {
	case <-runCtx.Done():
		elapsed := time.Since(start)
		c.transition(StateCancelled, inv, runCtx.Err().Error())
		if c.meter != nil {
			_ = c.meter.EmitToolCall(ctx, inv.ActorID, inv.OrgID, inv.ToolName, "cancelled_timeout", elapsed, map[string]interface{}{"beat_id": inv.BeatID})
		}
		return &ToolResult{ToolName: inv.ToolName, State: StateCancelled, LatencyMS: elapsed.Milliseconds()}, runCtx.Err()
	case res = <-done:
	}

	elapsed := time.Since(start)
	if res.err != nil {
		c.transition(StateFailed, inv, res.err.Error())
		status := "failed"
		if res.recovered {
			status = "panic_recovered"
		}
		if c.meter != nil {
			_ = c.meter.EmitToolCall(ctx, inv.ActorID, inv.OrgID, inv.ToolName, status, elapsed, map[string]interface{}{
				"beat_id":   inv.BeatID,
				"recovered": res.recovered,
			})
		}
		return &ToolResult{
			ToolName:   inv.ToolName,
			State:      StateFailed,
			LatencyMS:  elapsed.Milliseconds(),
			Recovered:  res.recovered,
			StackTrace: res.stack,
		}, res.err
	}

	c.transition(StateVerifying, inv, "tool completed, emitting telemetry")
	if c.meter != nil {
		_ = c.meter.EmitToolCall(ctx, inv.ActorID, inv.OrgID, inv.ToolName, "completed", elapsed, map[string]interface{}{"beat_id": inv.BeatID})
	}
	c.transition(StateCompleted, inv, "done")

	return &ToolResult{
		ToolName:  inv.ToolName,
		State:     StateCompleted,
		Output:    res.out,
		LatencyMS: elapsed.Milliseconds(),
	}, nil
}
