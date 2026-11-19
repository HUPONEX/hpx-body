package oasis

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/huponex/hpx-body/pkg/spine"
)

const (
	DefaultVerifyBaseURL   = "https://api.onyx.tel"
	StatusVerified         = "OASIS_VERIFIED"
	StatusPendingHandshake = "OASIS_PENDING_HANDSHAKE"
	StatusUnboundThreat    = "UNBOUND_THREAT"
	StatusRevoked          = "REVOKED_KILLSWITCH"
)

var ErrUnboundThreatBlocked = errors.New("ONBP egress guard blocked UNBOUND_THREAT target; TIC-Tactonyx gate approval required")

type VerifyResponse struct {
	Target                string `json:"target"`
	Status                string `json:"status"`
	TrustScore            int    `json:"trust_score"`
	QuarantineRecommended bool   `json:"quarantine_recommended"`
	CanonicalOTURN        string `json:"canonical_ot_urn,omitempty"`
	Reason                string `json:"reason,omitempty"`
}

type GateApproval struct {
	GateID     string    `json:"gate_id"`
	Target     string    `json:"target"`
	ApprovedBy string    `json:"approved_by"`
	ExpiresAt  time.Time `json:"expires_at"`
}

type EgressGuard struct {
	mu          sync.RWMutex
	verifyBase  string
	agentURN    string
	spine       *spine.Client
	httpClient  *http.Client
	approvals   map[string]GateApproval
	allowLocal  bool
	cacheTTL    time.Duration
	verifyCache map[string]cachedVerify
}

type cachedVerify struct {
	res       VerifyResponse
	expiresAt time.Time
}

func NewEgressGuard(verifyBaseURL, agentURN string, spineClient *spine.Client) *EgressGuard {
	base := strings.TrimRight(strings.TrimSpace(verifyBaseURL), "/")
	if base == "" {
		base = strings.TrimRight(strings.TrimSpace(os.Getenv("OASIS_VERIFY_URL")), "/")
	}
	if base == "" {
		base = DefaultVerifyBaseURL
	}
	return &EgressGuard{
		verifyBase:  base,
		agentURN:    agentURN,
		spine:       spineClient,
		httpClient:  &http.Client{Timeout: 5 * time.Second},
		approvals:   make(map[string]GateApproval),
		allowLocal:  true,
		cacheTTL:    2 * time.Minute,
		verifyCache: make(map[string]cachedVerify),
	}
}

// ApproveTarget records an explicit TIC-Tactonyx human approval override for an external target.
func (g *EgressGuard) ApproveTarget(approval GateApproval) {
	if g == nil {
		return
	}
	host := NormalizeTargetHost(approval.Target)
	if approval.ExpiresAt.IsZero() {
		approval.ExpiresAt = time.Now().UTC().Add(30 * time.Minute)
	}
	g.mu.Lock()
	g.approvals[host] = approval
	g.mu.Unlock()
}

func (g *EgressGuard) hasApproval(host string) bool {
	g.mu.RLock()
	ap, ok := g.approvals[host]
	g.mu.RUnlock()
	return ok && time.Now().UTC().Before(ap.ExpiresAt)
}

func NormalizeTargetHost(raw string) string {
	s := strings.TrimSpace(raw)
	if strings.Contains(s, "://") {
		if u, err := url.Parse(s); err == nil && u.Hostname() != "" {
			return strings.ToLower(u.Hostname())
		}
	}
	if h, _, err := net.SplitHostPort(s); err == nil && h != "" {
		return strings.ToLower(h)
	}
	return strings.TrimSuffix(strings.ToLower(s), ".")
}

// CheckEgress verifies an external tool/webhook URL or host against api.onyx.tel/v1/oasis/verify.
// If the destination is UNBOUND_THREAT and lacks an explicit TIC-Tactonyx approval, it emits
// `action: "beat.gate.requested"` to SPINE and returns ErrUnboundThreatBlocked.
func (g *EgressGuard) CheckEgress(ctx context.Context, rawTarget, beatID, actorID, orgID string) (*VerifyResponse, error) {
	if g == nil {
		return nil, nil
	}
	host := NormalizeTargetHost(rawTarget)
	if host == "" {
		return nil, errors.New("empty egress target")
	}

	if g.allowLocal && (host == "localhost" || host == "127.0.0.1" || host == "::1") {
		return &VerifyResponse{Target: host, Status: StatusVerified, TrustScore: 100, Reason: "loopback_allowed"}, nil
	}

	if g.hasApproval(host) {
		return &VerifyResponse{Target: host, Status: StatusVerified, TrustScore: 100, Reason: "tactonyx_gate_approved"}, nil
	}

	res, err := g.VerifyTarget(ctx, host)
	if err != nil {
		// Fail closed under zero-trust default posture
		res = &VerifyResponse{
			Target:                host,
			Status:                StatusUnboundThreat,
			TrustScore:            0,
			QuarantineRecommended: true,
			Reason:                fmt.Sprintf("oasis verification transport error: %v", err),
		}
	}

	if res.Status == StatusVerified {
		return res, nil
	}

	gateID := fmt.Sprintf("gate_egress_%d", time.Now().UnixNano())
	if g.spine != nil {
		_ = g.spine.PublishEnvelope("os.beat.gate.requested", "beat.gate.requested", host, actorID, orgID, map[string]interface{}{
			"gate_id":                gateID,
			"beat_id":                beatID,
			"agent_urn":              g.agentURN,
			"target":                 host,
			"oasis_status":           res.Status,
			"trust_score":            res.TrustScore,
			"quarantine_recommended": res.QuarantineRecommended,
			"reason":                 res.Reason,
			"options": []map[string]string{
				{"key_id": "approve", "label": "Approve Egress Override", "action": "tactonyx.decision.approve", "variant": "approve"},
				{"key_id": "reject", "label": "Block & Quarantine", "action": "tactonyx.decision.reject", "variant": "reject"},
			},
		})
	}

	return res, fmt.Errorf("%w: target=%s status=%s gate_id=%s", ErrUnboundThreatBlocked, host, res.Status, gateID)
}

func (g *EgressGuard) VerifyTarget(ctx context.Context, host string) (*VerifyResponse, error) {
	g.mu.RLock()
	if c, ok := g.verifyCache[host]; ok && time.Now().Before(c.expiresAt) {
		resCopy := c.res
		g.mu.RUnlock()
		return &resCopy, nil
	}
	g.mu.RUnlock()

	endpoint := fmt.Sprintf("%s/v1/oasis/verify?target=%s", g.verifyBase, url.QueryEscape(host))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	resp, err := g.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("oasis verify HTTP %d", resp.StatusCode)
	}

	var parsed VerifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, err
	}

	if parsed.Status == StatusVerified {
		g.mu.Lock()
		g.verifyCache[host] = cachedVerify{res: parsed, expiresAt: time.Now().Add(g.cacheTTL)}
		g.mu.Unlock()
	}
	return &parsed, nil
}
