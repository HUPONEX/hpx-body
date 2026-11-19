package cordis

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/huponex/hpx-body/pkg/oasis"
	"github.com/huponex/hpx-body/pkg/pips"
	"github.com/huponex/hpx-body/pkg/router"
	"github.com/huponex/hpx-body/pkg/secrets"
	"github.com/huponex/hpx-body/pkg/spine"
)

func TestChassisEgressGuardPanicRecoveryAndMetering(t *testing.T) {
	var mu sync.Mutex
	var emittedActions []string
	var pipsEvents []pips.TelemetryEvent

	// Mock Oasis ONBP verifier
	oasisSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		target := r.URL.Query().Get("target")
		w.Header().Set("Content-Type", "application/json")
		if target == "api.onyx.tel" {
			_ = json.NewEncoder(w).Encode(oasis.VerifyResponse{Target: target, Status: oasis.StatusVerified, TrustScore: 100})
			return
		}
		_ = json.NewEncoder(w).Encode(oasis.VerifyResponse{
			Target: target, Status: oasis.StatusUnboundThreat, TrustScore: 0, QuarantineRecommended: true,
		})
	}))
	defer oasisSrv.Close()

	// Mock PIPs telemetry endpoint
	pipsSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var ev pips.TelemetryEvent
		if err := json.NewDecoder(r.Body).Decode(&ev); err == nil {
			mu.Lock()
			pipsEvents = append(pipsEvents, ev)
			mu.Unlock()
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}))
	defer pipsSrv.Close()

	// Mock OpenRouter endpoint
	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"synthesized_ok"}}],"usage":{"prompt_tokens":42,"completion_tokens":18}}`))
	}))
	defer llmSrv.Close()

	spineClient := spine.NewMockClient("tel.onyx.hpx.runtime.body.001", func(subject string, env spine.SpineEnvelope) {
		mu.Lock()
		emittedActions = append(emittedActions, env.Action)
		mu.Unlock()
		if env.SourceURN != "ot.hpx.runtime.body.001" {
			t.Errorf("expected canonical ot.* source_urn, got %s", env.SourceURN)
		}
	})

	meter := pips.NewMeter(pipsSrv.URL, "ot.hpx.runtime.body.001")
	meter.SetSyncMode(true)

	sec := secrets.NewResolver(secrets.Config{})
	sec.SetInMemory("OPENROUTER_API_KEY", "sk-test-ram-only", time.Minute)

	rtr := router.NewRouter(sec, meter)
	rtr.SetEndpoint(router.ProviderOpenRouter, llmSrv.URL)

	guard := oasis.NewEgressGuard(oasisSrv.URL, "ot.hpx.runtime.body.001", spineClient)

	ch := NewChassis(ChassisConfig{
		AgentURN:       "tel.onyx.hpx.runtime.body.001",
		DefaultTimeout: 500 * time.Millisecond,
		Spine:          spineClient,
		Meter:          meter,
		Router:         rtr,
		EgressGuard:    guard,
	})

	_ = ch.RegisterTool(ToolFunc{
		ToolName: "webhook_call",
		Desc:     "Calls external webhook",
		Fn: func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
			return map[string]interface{}{"ok": true}, nil
		},
	})
	_ = ch.RegisterTool(ToolFunc{
		ToolName: "panic_tool",
		Desc:     "Intentionally panics",
		Fn: func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
			panic("simulated crash")
		},
	})
	_ = ch.RegisterTool(ToolFunc{
		ToolName: "slow_tool",
		Desc:     "Exceeds timeout",
		Fn: func(ctx context.Context, params map[string]interface{}) (map[string]interface{}, error) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(2 * time.Second):
				return nil, nil
			}
		},
	})

	ctx := context.Background()

	// 1. Unbound threat blocked & triggers beat.gate.requested
	res, err := ch.ExecuteTool(ctx, ToolInvocation{
		BeatID: "BEAT-7.2", ActorID: "bill_mack", OrgID: "hpi",
		ToolName: "webhook_call", ExternalTarget: "https://example.com/hook",
	})
	if !errors.Is(err, oasis.ErrUnboundThreatBlocked) || res.State != StateGateRequested {
		t.Fatalf("expected ErrUnboundThreatBlocked, got state=%s err=%v", res.State, err)
	}

	// 2. Explicit TIC-Tactonyx approval unblocks target
	guard.ApproveTarget(oasis.GateApproval{GateID: "gate-1", Target: "example.com", ApprovedBy: "bill_mack"})
	res, err = ch.ExecuteTool(ctx, ToolInvocation{
		BeatID: "BEAT-7.2", ActorID: "bill_mack", OrgID: "hpi",
		ToolName: "webhook_call", ExternalTarget: "https://example.com/hook",
	})
	if err != nil || res.State != StateCompleted {
		t.Fatalf("expected approved tool execution to succeed, got state=%s err=%v", res.State, err)
	}

	// 3. Structured panic recovery
	res, err = ch.ExecuteTool(ctx, ToolInvocation{BeatID: "BEAT-7.2", ToolName: "panic_tool"})
	if err == nil || !res.Recovered || res.State != StateFailed {
		t.Fatalf("expected panic recovery, got res=%+v err=%v", res, err)
	}

	// 4. Timeout cancellation lifecycle
	res, err = ch.ExecuteTool(ctx, ToolInvocation{BeatID: "BEAT-7.2", ToolName: "slow_tool", Timeout: 40 * time.Millisecond})
	if !errors.Is(err, context.DeadlineExceeded) || res.State != StateCancelled {
		t.Fatalf("expected timeout cancellation, got state=%s err=%v", res.State, err)
	}

	// 5. BYOCoSA+ Provider Router + PIPs metering
	inf, err := ch.Router().Complete(ctx, router.InferenceRequest{
		Provider: router.ProviderOpenRouter,
		Model:    "anthropic/claude-3.5-sonnet",
		Messages: []router.Message{{Role: "user", Content: "ping"}},
		ActorID:  "bill_mack",
		OrgID:    "hpi",
	})
	if err != nil || inf.Content != "synthesized_ok" || inf.PromptTokens != 42 {
		t.Fatalf("unexpected inference result: %+v err=%v", inf, err)
	}

	mu.Lock()
	defer mu.Unlock()
	foundGate := false
	for _, a := range emittedActions {
		if a == "beat.gate.requested" {
			foundGate = true
			break
		}
	}
	if !foundGate {
		t.Fatalf("expected beat.gate.requested on SPINE, saw %v", emittedActions)
	}
	if len(pipsEvents) < 5 {
		t.Fatalf("expected >=5 PIPs metered events (tools + inference), got %d", len(pipsEvents))
	}
}
