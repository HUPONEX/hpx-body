package spine

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/huponex/hpx-body/pkg/urn"
	"github.com/nats-io/nats.go"
)

type SpineEnvelope struct {
	EventID   string                 `json:"event_id"`
	Timestamp string                 `json:"timestamp"`
	SourceURN string                 `json:"source_urn"`
	TargetURN string                 `json:"target_urn,omitempty"`
	ActorID   string                 `json:"actor_id,omitempty"`
	OrgID     string                 `json:"org_id,omitempty"`
	Action    string                 `json:"action"`
	Economics map[string]interface{} `json:"economics,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type Client struct {
	nc        *nats.Conn
	js        nats.JetStreamContext
	sourceURN string
	manifest  []byte
	onPublish func(subject string, env SpineEnvelope) // optional test/local hook
}

func NewMockClient(sourceURN string, onPublish func(subject string, env SpineEnvelope)) *Client {
	return &Client{
		sourceURN: urn.CanonicalOrRaw(sourceURN),
		onPublish: onPublish,
	}
}

func Connect(natsURL, sourceURN string, manifest []byte) *Client {
	urlStr := strings.TrimSpace(natsURL)
	if urlStr == "" {
		urlStr = strings.TrimSpace(os.Getenv("NATS_URL"))
	}
	if urlStr == "" {
		urlStr = strings.TrimSpace(os.Getenv("SPINE_NATS_URL"))
	}
	if urlStr == "" {
		urlStr = nats.DefaultURL
	}

	canonicalSrc := urn.CanonicalOrRaw(sourceURN)
	if canonicalSrc == "" {
		canonicalSrc = urn.BodyURN
	}

	c := &Client{
		sourceURN: canonicalSrc,
		manifest:  manifest,
	}

	nc, err := nats.Connect(
		urlStr,
		nats.Name(canonicalSrc),
		nats.MaxReconnects(-1),
		nats.ReconnectWait(3*time.Second),
		nats.RetryOnFailedConnect(true),
		nats.ReconnectHandler(func(conn *nats.Conn) {
			log.Printf("[body-spine] 🔄 Reconnected to NATS (%s), re-registering %s...", conn.ConnectedUrl(), canonicalSrc)
			c.RegisterSnaponyx()
		}),
	)
	if err != nil {
		log.Printf("[body-spine] ⚠️ NATS connect warning (%v) — operating in offline mode", err)
		return c
	}

	c.nc = nc
	if js, jsErr := nc.JetStream(); jsErr == nil {
		c.js = js
	}
	if nc.IsConnected() && len(manifest) > 0 {
		c.RegisterSnaponyx()
	}
	return c
}

func (c *Client) IsConnected() bool {
	return c != nil && c.nc != nil && c.nc.IsConnected()
}

func (c *Client) ConnectedServer() string {
	if c.IsConnected() {
		return c.nc.ConnectedUrl()
	}
	return "disconnected"
}

func (c *Client) NATS() *nats.Conn {
	if c == nil {
		return nil
	}
	return c.nc
}

func (c *Client) JetStream() nats.JetStreamContext {
	if c == nil {
		return nil
	}
	return c.js
}

func (c *Client) Close() {
	if c != nil && c.nc != nil {
		_ = c.nc.Drain()
	}
}

func (c *Client) RegisterSnaponyx() {
	if !c.IsConnected() || len(c.manifest) == 0 {
		return
	}
	msg, err := c.nc.Request("os.register", c.manifest, 4*time.Second)
	if err != nil {
		log.Printf("[body-spine] ⚠️ SnapOnyx registration pending for %s: %v", c.sourceURN, err)
		return
	}
	log.Printf("[body-spine] ✅ SNAP-ON REGISTERED with SPINE (%s). Ack: %s", c.sourceURN, string(msg.Data))
}

func (c *Client) EnsureStream(name string, subjects []string) error {
	if c == nil || c.js == nil {
		return nil
	}
	if _, err := c.js.StreamInfo(name); err == nats.ErrStreamNotFound {
		_, err = c.js.AddStream(&nats.StreamConfig{
			Name:      name,
			Subjects:  subjects,
			Retention: nats.LimitsPolicy,
			Storage:   nats.FileStorage,
			MaxAge:    7 * 24 * time.Hour,
		})
		return err
	}
	return nil
}

func (c *Client) PublishEnvelope(subject, action, targetURN, actorID, orgID string, metadata map[string]interface{}) error {
	if c == nil {
		return nil
	}
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	env := SpineEnvelope{
		EventID:   fmt.Sprintf("evt_%s_%d", hex.EncodeToString(b), time.Now().UnixNano()),
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		SourceURN: urn.CanonicalOrRaw(c.sourceURN),
		TargetURN: urn.CanonicalOrRaw(targetURN),
		ActorID:   actorID,
		OrgID:     orgID,
		Action:    action,
		Economics: map[string]interface{}{"cost": 1, "currency": "pip"},
		Metadata:  metadata,
	}

	if c.onPublish != nil {
		c.onPublish(subject, env)
	}
	if !c.IsConnected() {
		return nil
	}
	payload, err := json.Marshal(env)
	if err != nil {
		return err
	}
	if c.js != nil {
		if _, err := c.js.Publish(subject, payload); err == nil {
			return nil
		}
	}
	return c.nc.Publish(subject, payload)
}
