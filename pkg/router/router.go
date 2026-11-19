package router

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/huponex/hpx-body/pkg/pips"
	"github.com/huponex/hpx-body/pkg/secrets"
)

type ProviderName string

const (
	ProviderOpenRouter ProviderName = "openrouter"
	ProviderGemini     ProviderName = "gemini"
	ProviderAnthropic  ProviderName = "anthropic"
	ProviderDeepSeek   ProviderName = "deepseek"
	ProviderOllama     ProviderName = "ollama"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type InferenceRequest struct {
	Provider      ProviderName `json:"provider"`
	Model         string       `json:"model"`
	Messages      []Message    `json:"messages"`
	Temperature   float64      `json:"temperature,omitempty"`
	MaxTokens     int          `json:"max_tokens,omitempty"`
	SecretPointer string       `json:"secret_pointer,omitempty"` // Optional override pointer e.g. infisical://prod/byocosa/KEY
	ActorID       string       `json:"actor_id,omitempty"`
	OrgID         string       `json:"org_id,omitempty"`
}

type InferenceResponse struct {
	Provider         ProviderName  `json:"provider"`
	Model            string        `json:"model"`
	Content          string        `json:"content"`
	PromptTokens     int           `json:"prompt_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
	Latency          time.Duration `json:"latency_ns"`
	CostUSD          float64       `json:"cost_usd"`
}

type EmbedRequest struct {
	Provider      ProviderName `json:"provider,omitempty"`
	Model         string       `json:"model,omitempty"`
	Input         string       `json:"input"`
	Dimensions    int          `json:"dimensions,omitempty"`
	SecretPointer string       `json:"secret_pointer,omitempty"`
	ActorID       string       `json:"actor_id,omitempty"`
	OrgID         string       `json:"org_id,omitempty"`
}

type Router struct {
	secrets    *secrets.Resolver
	meter      *pips.Meter
	httpClient *http.Client
	endpoints  map[ProviderName]string
}

func NewRouter(resolver *secrets.Resolver, meter *pips.Meter) *Router {
	if resolver == nil {
		resolver = secrets.NewResolver(secrets.Config{})
	}
	ollamaHost := strings.TrimRight(strings.TrimSpace(os.Getenv("OLLAMA_HOST")), "/")
	if ollamaHost == "" {
		ollamaHost = "http://127.0.0.1:11434"
	}
	return &Router{
		secrets:    resolver,
		meter:      meter,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		endpoints: map[ProviderName]string{
			ProviderOpenRouter: "https://openrouter.ai/api/v1/chat/completions",
			ProviderGemini:     "https://generativelanguage.googleapis.com/v1beta/models",
			ProviderAnthropic:  "https://api.anthropic.com/v1/messages",
			ProviderDeepSeek:   "https://api.deepseek.com/v1/chat/completions",
			ProviderOllama:     ollamaHost + "/api/chat",
		},
	}
}

func (r *Router) SetEndpoint(provider ProviderName, endpointURL string) {
	r.endpoints[provider] = strings.TrimRight(endpointURL, "/")
}

func (r *Router) Complete(ctx context.Context, req InferenceRequest) (*InferenceResponse, error) {
	if req.Provider == "" {
		req.Provider = ProviderOpenRouter
	}
	if len(req.Messages) == 0 {
		return nil, errors.New("at least one message is required")
	}

	start := time.Now()
	var res *InferenceResponse
	var err error

	switch req.Provider {
	case ProviderOpenRouter:
		res, err = r.callOpenAICompat(ctx, req, "OPENROUTER_API_KEY", "anthropic/claude-3.5-sonnet")
	case ProviderDeepSeek:
		res, err = r.callOpenAICompat(ctx, req, "DEEPSEEK_API_KEY", "deepseek-chat")
	case ProviderAnthropic:
		res, err = r.callAnthropic(ctx, req)
	case ProviderGemini:
		res, err = r.callGemini(ctx, req)
	case ProviderOllama:
		res, err = r.callOllama(ctx, req)
	default:
		return nil, fmt.Errorf("unsupported BYOCoSA+ provider %q", req.Provider)
	}
	if err != nil {
		return nil, err
	}

	res.Latency = time.Since(start)
	res.CostUSD = estimateCostUSD(res.Provider, res.PromptTokens, res.CompletionTokens)

	if r.meter != nil {
		_ = r.meter.EmitInference(
			ctx, req.ActorID, req.OrgID, string(res.Provider), res.Model,
			res.PromptTokens, res.CompletionTokens, res.Latency, res.CostUSD,
			map[string]interface{}{"router": "byocosa_plus"},
		)
	}
	return res, nil
}

// Embed generates a normalized vector (default 1536-dim for Supabase pgvector).
// Uses Gemini or OpenRouter when a key is present in RAM, falling back to a deterministic
// unit-normalized trigram projection so offline/test Crucibles remain reproducible.
func (r *Router) Embed(ctx context.Context, req EmbedRequest) ([]float32, error) {
	dims := req.Dimensions
	if dims <= 0 {
		dims = 1536
	}
	text := strings.TrimSpace(req.Input)
	if text == "" {
		return nil, errors.New("embedding input is required")
	}

	start := time.Now()
	provider := req.Provider
	if provider == "" {
		provider = ProviderGemini
	}

	// Attempt live Gemini embedding if key is available in RAM
	if provider == ProviderGemini {
		keyName := req.SecretPointer
		if keyName == "" {
			keyName = "GEMINI_API_KEY"
		}
		if apiKey, err := r.secrets.Resolve(ctx, keyName); err == nil && apiKey != "" {
			model := req.Model
			if model == "" {
				model = "text-embedding-004"
			}
			vec, embErr := r.embedGemini(ctx, apiKey, model, text, dims)
			if embErr == nil && len(vec) == dims {
				if r.meter != nil {
					_ = r.meter.EmitInference(ctx, req.ActorID, req.OrgID, string(provider), model, len(text)/4+1, 0, time.Since(start), 0.0001, map[string]interface{}{"op": "embed"})
				}
				return vec, nil
			}
		}
	}

	// Deterministic local unit-normalized projection (zero external network dependency)
	vec := DeterministicEmbedding(text, dims)
	if r.meter != nil {
		_ = r.meter.EmitInference(ctx, req.ActorID, req.OrgID, "local_deterministic", "body-embed-v1", len(text)/4+1, 0, time.Since(start), 0, map[string]interface{}{"op": "embed_fallback"})
	}
	return vec, nil
}

func (r *Router) embedGemini(ctx context.Context, apiKey, model, text string, dims int) ([]float32, error) {
	ep := fmt.Sprintf("%s/%s:embedContent?key=%s", r.endpoints[ProviderGemini], model, apiKey)
	body, _ := json.Marshal(map[string]interface{}{
		"model":                "models/" + model,
		"content":              map[string]interface{}{"parts": []map[string]string{{"text": text}}},
		"outputDimensionality": dims,
	})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, ep, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini embed HTTP %d", resp.StatusCode)
	}
	var out struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Embedding.Values, nil
}

// DeterministicEmbedding maps text tokens into a normalized float32 vector of length dims.
func DeterministicEmbedding(text string, dims int) []float32 {
	vec := make([]float32, dims)
	tokens := strings.Fields(strings.ToLower(text))
	if len(tokens) == 0 {
		tokens = []string{"empty"}
	}
	for _, tok := range tokens {
		sum := sha256.Sum256([]byte("vectonyx:" + tok))
		for i := 0; i < 8; i++ {
			idx := int(binary.BigEndian.Uint16(sum[i*2:i*2+2])) % dims
			sign := float32(1.0)
			if sum[(i+16)%32]%2 == 1 {
				sign = -1.0
			}
			vec[idx] += sign
		}
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		inv := float32(1.0 / math.Sqrt(norm))
		for i := range vec {
			vec[i] *= inv
		}
	}
	return vec
}

func (r *Router) resolveKey(ctx context.Context, pointer, defaultEnv string) (string, error) {
	if strings.TrimSpace(pointer) != "" {
		return r.secrets.Resolve(ctx, pointer)
	}
	return r.secrets.Resolve(ctx, defaultEnv)
}

func (r *Router) callOpenAICompat(ctx context.Context, req InferenceRequest, defaultEnv, defaultModel string) (*InferenceResponse, error) {
	apiKey, err := r.resolveKey(ctx, req.SecretPointer, defaultEnv)
	if err != nil {
		return nil, err
	}
	model := req.Model
	if model == "" {
		model = defaultModel
	}
	payload := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
	}
	if req.MaxTokens > 0 {
		payload["max_tokens"] = req.MaxTokens
	}
	if req.Temperature > 0 {
		payload["temperature"] = req.Temperature
	}
	b, _ := json.Marshal(payload)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoints[req.Provider], bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	if req.Provider == ProviderOpenRouter {
		httpReq.Header.Set("HTTP-Referer", "https://ethos.hpn.ngo")
		httpReq.Header.Set("X-Title", "huponex-body-v1")
	}

	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s HTTP %d: %s", req.Provider, resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var parsed struct {
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	if len(parsed.Choices) == 0 {
		return nil, fmt.Errorf("%s returned empty choices", req.Provider)
	}
	return &InferenceResponse{
		Provider:         req.Provider,
		Model:            model,
		Content:          parsed.Choices[0].Message.Content,
		PromptTokens:     parsed.Usage.PromptTokens,
		CompletionTokens: parsed.Usage.CompletionTokens,
	}, nil
}

func (r *Router) callAnthropic(ctx context.Context, req InferenceRequest) (*InferenceResponse, error) {
	apiKey, err := r.resolveKey(ctx, req.SecretPointer, "ANTHROPIC_API_KEY")
	if err != nil {
		return nil, err
	}
	model := req.Model
	if model == "" {
		model = "claude-3-5-sonnet-latest"
	}
	maxTok := req.MaxTokens
	if maxTok <= 0 {
		maxTok = 1024
	}
	var sysPrompt string
	var msgs []Message
	for _, m := range req.Messages {
		if m.Role == "system" {
			sysPrompt += m.Content + "\n"
		} else {
			msgs = append(msgs, m)
		}
	}
	payload := map[string]interface{}{
		"model":      model,
		"max_tokens": maxTok,
		"messages":   msgs,
	}
	if sysPrompt != "" {
		payload["system"] = strings.TrimSpace(sysPrompt)
	}
	b, _ := json.Marshal(payload)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoints[ProviderAnthropic], bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anthropic HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	outText := ""
	if len(parsed.Content) > 0 {
		outText = parsed.Content[0].Text
	}
	return &InferenceResponse{
		Provider:         ProviderAnthropic,
		Model:            model,
		Content:          outText,
		PromptTokens:     parsed.Usage.InputTokens,
		CompletionTokens: parsed.Usage.OutputTokens,
	}, nil
}

func (r *Router) callGemini(ctx context.Context, req InferenceRequest) (*InferenceResponse, error) {
	apiKey, err := r.resolveKey(ctx, req.SecretPointer, "GEMINI_API_KEY")
	if err != nil {
		return nil, err
	}
	model := req.Model
	if model == "" {
		model = "gemini-2.5-flash"
	}
	var contents []map[string]interface{}
	for _, m := range req.Messages {
		role := "user"
		if m.Role == "assistant" {
			role = "model"
		}
		contents = append(contents, map[string]interface{}{
			"role":  role,
			"parts": []map[string]string{{"text": m.Content}},
		})
	}
	b, _ := json.Marshal(map[string]interface{}{"contents": contents})
	ep := fmt.Sprintf("%s/%s:generateContent?key=%s", r.endpoints[ProviderGemini], model, apiKey)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, ep, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	outText := ""
	if len(parsed.Candidates) > 0 && len(parsed.Candidates[0].Content.Parts) > 0 {
		outText = parsed.Candidates[0].Content.Parts[0].Text
	}
	return &InferenceResponse{
		Provider:         ProviderGemini,
		Model:            model,
		Content:          outText,
		PromptTokens:     parsed.UsageMetadata.PromptTokenCount,
		CompletionTokens: parsed.UsageMetadata.CandidatesTokenCount,
	}, nil
}

func (r *Router) callOllama(ctx context.Context, req InferenceRequest) (*InferenceResponse, error) {
	model := req.Model
	if model == "" {
		model = "llama3.2"
	}
	b, _ := json.Marshal(map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
		"stream":   false,
	})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, r.endpoints[ProviderOllama], bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var parsed struct {
		Message         Message `json:"message"`
		PromptEvalCount int     `json:"prompt_eval_count"`
		EvalCount       int     `json:"eval_count"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	return &InferenceResponse{
		Provider:         ProviderOllama,
		Model:            model,
		Content:          parsed.Message.Content,
		PromptTokens:     parsed.PromptEvalCount,
		CompletionTokens: parsed.EvalCount,
	}, nil
}

func estimateCostUSD(p ProviderName, promptTok, compTok int) float64 {
	if p == ProviderOllama {
		return 0
	}
	// Conservative blended USD estimate per 1M tokens
	return (float64(promptTok)*0.50 + float64(compTok)*1.50) / 1_000_000.0
}
