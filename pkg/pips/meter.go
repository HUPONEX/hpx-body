package pips

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/huponex/hpx-body/pkg/urn"
)

const DefaultEmitEndpoint = "https://api.pips.tel/api/v1/telemetry/emit"

type TelemetryEvent struct {
	SourceURN        string                 `json:"source_urn"`
	Action           string                 `json:"action"`
	Subject          string                 `json:"subject,omitempty"`
	ActorID          string                 `json:"actor_id,omitempty"`
	OrgID            string                 `json:"org_id,omitempty"`
	TargetID         string                 `json:"target_id,omitempty"`
	Model            string                 `json:"model,omitempty"`
	PromptTokens     int                    `json:"prompt_tokens,omitempty"`
	CompletionTokens int                    `json:"completion_tokens,omitempty"`
	Tags             map[string]interface{} `json:"tags,omitempty"`
	Economics        map[string]interface{} `json:"economics,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

type Meter struct {
	endpoint   string
	sourceURN  string
	httpClient *http.Client
	syncMode   bool
}

func NewMeter(endpoint, sourceURN string) *Meter {
	ep := strings.TrimSpace(endpoint)
	if ep == "" {
		ep = strings.TrimSpace(os.Getenv("PIPS_TELEMETRY_URL"))
	}
	if ep == "" {
		if base := strings.TrimRight(strings.TrimSpace(os.Getenv("PIPS_API_URL")), "/"); base != "" {
			ep = base + "/api/v1/telemetry/emit"
		} else {
			ep = DefaultEmitEndpoint
		}
	}
	src := urn.CanonicalOrRaw(sourceURN)
	if src == "" {
		src = urn.BodyURN
	}
	return &Meter{
		endpoint:   ep,
		sourceURN:  src,
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}
}

// SetSyncMode makes Emit block on HTTP delivery (useful in unit tests and CLI probes).
func (m *Meter) SetSyncMode(sync bool) {
	if m != nil {
		m.syncMode = sync
	}
}

func (m *Meter) Emit(ctx context.Context, ev TelemetryEvent) error {
	if m == nil || m.endpoint == "" {
		return nil
	}
	if ev.SourceURN == "" {
		ev.SourceURN = m.sourceURN
	} else {
		ev.SourceURN = urn.CanonicalOrRaw(ev.SourceURN)
	}
	if ev.Economics == nil {
		ev.Economics = map[string]interface{}{"cost": 1, "currency": "pip"}
	}

	send := func(sendCtx context.Context) error {
		b, err := json.Marshal(ev)
		if err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(sendCtx, http.MethodPost, m.endpoint, bytes.NewReader(b))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := m.httpClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			return fmt.Errorf("pips-api emit returned HTTP %d", resp.StatusCode)
		}
		return nil
	}

	if m.syncMode {
		return send(ctx)
	}
	go func() {
		bgCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = send(bgCtx)
	}()
	return nil
}

func (m *Meter) EmitInference(ctx context.Context, actorID, orgID, provider, model string, promptTok, compTok int, latency time.Duration, costUSD float64, meta map[string]interface{}) error {
	if meta == nil {
		meta = map[string]interface{}{}
	}
	meta["provider"] = provider
	meta["latency_ms"] = latency.Milliseconds()
	return m.Emit(ctx, TelemetryEvent{
		SourceURN:        m.sourceURN,
		Action:           "agent.inference.completed",
		Subject:          "os.agent.inference",
		ActorID:          actorID,
		OrgID:            orgID,
		Model:            model,
		PromptTokens:     promptTok,
		CompletionTokens: compTok,
		Economics: map[string]interface{}{
			"cost":              1,
			"currency":          "pip",
			"cost_usd":          costUSD,
			"model":             model,
			"prompt_tokens":     promptTok,
			"completion_tokens": compTok,
			"latency_ms":        latency.Milliseconds(),
		},
		Metadata: meta,
	})
}

func (m *Meter) EmitToolCall(ctx context.Context, actorID, orgID, toolName, status string, latency time.Duration, meta map[string]interface{}) error {
	if meta == nil {
		meta = map[string]interface{}{}
	}
	meta["tool_name"] = toolName
	meta["status"] = status
	meta["latency_ms"] = latency.Milliseconds()
	return m.Emit(ctx, TelemetryEvent{
		SourceURN: m.sourceURN,
		Action:    "agent.tool.executed",
		Subject:   "os.agent.tool",
		ActorID:   actorID,
		OrgID:     orgID,
		Tags:      map[string]interface{}{"tool": toolName, "status": status},
		Economics: map[string]interface{}{
			"cost":       1,
			"currency":   "pip",
			"latency_ms": latency.Milliseconds(),
		},
		Metadata: meta,
	})
}
