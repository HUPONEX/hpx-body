package secrets

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

const DefaultInfisicalURL = "https://enigma.tinc.tel"

type cachedSecret struct {
	value     string
	expiresAt time.Time
}

type Resolver struct {
	mu           sync.RWMutex
	baseURL      string
	projectID    string
	environment  string
	clientID     string
	clientSecret string
	accessToken  string
	tokenExpiry  time.Time
	cacheTTL     time.Duration
	cache        map[string]cachedSecret
	httpClient   *http.Client
}

type Config struct {
	BaseURL      string
	ProjectID    string
	Environment  string
	ClientID     string
	ClientSecret string
	CacheTTL     time.Duration
}

func NewResolver(cfg Config) *Resolver {
	base := strings.TrimRight(firstNonEmpty(cfg.BaseURL, os.Getenv("INFISICAL_API_URL"), DefaultInfisicalURL), "/")
	ttl := cfg.CacheTTL
	if ttl <= 0 {
		ttl = 5 * time.Minute
	}
	return &Resolver{
		baseURL:      base,
		projectID:    firstNonEmpty(cfg.ProjectID, os.Getenv("INFISICAL_PROJECT_ID")),
		environment:  firstNonEmpty(cfg.Environment, os.Getenv("INFISICAL_ENV"), "prod"),
		clientID:     firstNonEmpty(cfg.ClientID, os.Getenv("INFISICAL_CLIENT_ID")),
		clientSecret: firstNonEmpty(cfg.ClientSecret, os.Getenv("INFISICAL_CLIENT_SECRET")),
		cacheTTL:     ttl,
		cache:        make(map[string]cachedSecret),
		httpClient:   &http.Client{Timeout: 10 * time.Second},
	}
}

// Resolve reads a secret strictly in RAM. It supports:
//   - Direct environment variables injected by `infisical run --env=prod`
//   - Pointer URIs: `infisical://<env>/<path>/<KEY>` (e.g. `infisical://prod/byocosa/OPENROUTER_API_KEY`)
// It NEVER reads or writes `.env` files on disk.
func (r *Resolver) Resolve(ctx context.Context, keyOrPointer string) (string, error) {
	raw := strings.TrimSpace(keyOrPointer)
	if raw == "" {
		return "", errors.New("empty secret key or pointer")
	}

	// 1. Check RAM TTL cache first (supports in-memory BYOCoSA+ session overrides)
	r.mu.RLock()
	if c, ok := r.cache[raw]; ok && time.Now().Before(c.expiresAt) {
		r.mu.RUnlock()
		return c.value, nil
	}
	r.mu.RUnlock()

	// 2. Direct RAM env lookup (standard `infisical run` injection)
	if !strings.HasPrefix(raw, "infisical://") {
		if v := strings.TrimSpace(os.Getenv(raw)); v != "" {
			if strings.HasPrefix(v, "infisical://") {
				raw = v
			} else {
				return v, nil
			}
		} else if r.clientID == "" || r.clientSecret == "" || r.projectID == "" {
			return "", fmt.Errorf("secret %q not present in RAM environment and Infisical Universal Auth is unconfigured", raw)
		}
	}

	envName := r.environment
	secretPath := "/"
	secretKey := raw

	if strings.HasPrefix(raw, "infisical://") {
		trimmed := strings.TrimPrefix(raw, "infisical://")
		parts := strings.Split(trimmed, "/")
		if len(parts) < 2 {
			return "", fmt.Errorf("invalid infisical pointer %q: expected infisical://<env>/<path>/<KEY>", raw)
		}
		envName = parts[0]
		secretKey = parts[len(parts)-1]
		if len(parts) > 2 {
			secretPath = "/" + strings.Join(parts[1:len(parts)-1], "/")
		}
	}

	// Allow env override matching the final key name when running under `infisical run`
	if v := strings.TrimSpace(os.Getenv(secretKey)); v != "" && !strings.HasPrefix(v, "infisical://") {
		return v, nil
	}

	val, err := r.fetchFromInfisical(ctx, envName, secretPath, secretKey)
	if err != nil {
		return "", err
	}

	r.mu.Lock()
	r.cache[raw] = cachedSecret{value: val, expiresAt: time.Now().Add(r.cacheTTL)}
	r.mu.Unlock()
	return val, nil
}

// SetInMemory injects a secret strictly into the process RAM cache (useful for BYOCoSA+ session overrides & tests).
func (r *Resolver) SetInMemory(key, val string, ttl time.Duration) {
	if ttl <= 0 {
		ttl = r.cacheTTL
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cache[key] = cachedSecret{value: val, expiresAt: time.Now().Add(ttl)}
}

// PurgeRAM clears all cached secrets and bearer tokens from memory.
func (r *Resolver) PurgeRAM() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for k := range r.cache {
		delete(r.cache, k)
	}
	r.accessToken = ""
	r.tokenExpiry = time.Time{}
}

func (r *Resolver) fetchFromInfisical(ctx context.Context, envName, secretPath, secretKey string) (string, error) {
	if r.clientID == "" || r.clientSecret == "" || r.projectID == "" {
		return "", fmt.Errorf("cannot resolve %q from %s: INFISICAL_CLIENT_ID, INFISICAL_CLIENT_SECRET, and INFISICAL_PROJECT_ID required", secretKey, r.baseURL)
	}

	tok, err := r.ensureAccessToken(ctx)
	if err != nil {
		return "", err
	}

	endpoint := fmt.Sprintf("%s/api/v3/secrets/raw/%s?workspaceId=%s&environment=%s&secretPath=%s",
		r.baseURL,
		url.PathEscape(secretKey),
		url.QueryEscape(r.projectID),
		url.QueryEscape(envName),
		url.QueryEscape(secretPath),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+tok)

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("infisical fetch %q: %w", secretKey, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("infisical returned HTTP %d for %q: %s", resp.StatusCode, secretKey, strings.TrimSpace(string(body)))
	}

	var parsed struct {
		Secret struct {
			SecretValue string `json:"secretValue"`
		} `json:"secret"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", err
	}
	if parsed.Secret.SecretValue == "" {
		return "", fmt.Errorf("empty secret value returned for %q", secretKey)
	}
	return parsed.Secret.SecretValue, nil
}

func (r *Resolver) ensureAccessToken(ctx context.Context) (string, error) {
	r.mu.RLock()
	if r.accessToken != "" && time.Now().Before(r.tokenExpiry) {
		tok := r.accessToken
		r.mu.RUnlock()
		return tok, nil
	}
	r.mu.RUnlock()

	payload, _ := json.Marshal(map[string]string{
		"clientId":     r.clientID,
		"clientSecret": r.clientSecret,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.baseURL+"/api/v1/auth/universal-auth/login", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("infisical universal-auth login failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("infisical universal-auth login HTTP %d", resp.StatusCode)
	}

	var authRes struct {
		AccessToken string `json:"accessToken"`
		ExpiresIn   int    `json:"expiresIn"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&authRes); err != nil {
		return "", err
	}
	if authRes.AccessToken == "" {
		return "", errors.New("infisical returned empty accessToken")
	}

	exp := time.Duration(authRes.ExpiresIn-30) * time.Second
	if exp <= 0 {
		exp = 10 * time.Minute
	}
	r.mu.Lock()
	r.accessToken = authRes.AccessToken
	r.tokenExpiry = time.Now().Add(exp)
	r.mu.Unlock()
	return authRes.AccessToken, nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
