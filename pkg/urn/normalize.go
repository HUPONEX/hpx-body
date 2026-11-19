package urn

import (
	"fmt"
	"regexp"
	"strings"
)

const BodyURN = "ot.hpx.runtime.body.001"

var segmentSanitizer = regexp.MustCompile(`[^a-z0-9_-]+`)

var FederatedTLDs = map[string]bool{
	".onyx": true, ".ot": true, ".oasis": true, ".oas": true,
	".pearl": true, ".ruby": true, ".opal": true, ".amber": true,
	".coral": true, ".topaz": true, ".jade": true, ".quartz": true,
	".sapphire": true, ".garnet": true, ".zenith": true,
}

type Identity struct {
	CanonicalOT   string `json:"canonical_ot_urn"`
	LegacyTelOnyx string `json:"legacy_urn"`
	OrgCode       string `json:"org_code"`
	Layer         string `json:"layer"`
	Service       string `json:"service"`
	InstanceID    string `json:"instance_id"`
}

func SanitizeSegment(val, fallback string) string {
	v := strings.ToLower(strings.TrimSpace(val))
	v = segmentSanitizer.ReplaceAllString(v, "")
	if v == "" {
		return fallback
	}
	return v
}

func Build(org, layer, service, instance string) Identity {
	o := SanitizeSegment(org, "hpx")
	l := SanitizeSegment(layer, "runtime")
	s := SanitizeSegment(service, "body")
	i := SanitizeSegment(instance, "001")
	return Identity{
		CanonicalOT:   fmt.Sprintf("ot.%s.%s.%s.%s", o, l, s, i),
		LegacyTelOnyx: fmt.Sprintf("tel.onyx.%s.%s.%s.%s", o, l, s, i),
		OrgCode:       o,
		Layer:         l,
		Service:       s,
		InstanceID:    i,
	}
}

// Normalize converts ot.<org>.<layer>.<svc>.<id>, legacy tel.onyx.<org>.<layer>.<svc>.<id>,
// or federated TLD roots into a canonical Identity stamped with ot.*.
func Normalize(raw string) (Identity, error) {
	clean := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(raw)), ".")
	if clean == "" {
		return Identity{}, fmt.Errorf("empty URN")
	}
	parts := strings.Split(clean, ".")

	if len(parts) == 5 && parts[0] == "ot" {
		return Build(parts[1], parts[2], parts[3], parts[4]), nil
	}
	if len(parts) == 6 && parts[0] == "tel" && parts[1] == "onyx" {
		return Build(parts[2], parts[3], parts[4], parts[5]), nil
	}

	if idx := strings.LastIndex(clean, "."); idx > 0 {
		if FederatedTLDs[clean[idx:]] {
			labels := parts[:len(parts)-1]
			switch len(labels) {
			case 1:
				return Build(labels[0], "node", "root", "001"), nil
			case 2:
				return Build(labels[1], "node", labels[0], "001"), nil
			case 3:
				return Build(labels[2], labels[1], labels[0], "001"), nil
			default:
				n := len(labels)
				return Build(labels[n-1], labels[n-2], labels[n-3], labels[n-4]), nil
			}
		}
	}

	return Identity{}, fmt.Errorf("invalid URN %q: expected ot.* or tel.onyx.*", raw)
}

// CanonicalOrRaw returns the normalized ot.* URN if parseable, otherwise trims and returns input.
func CanonicalOrRaw(raw string) string {
	if id, err := Normalize(raw); err == nil {
		return id.CanonicalOT
	}
	return strings.TrimSpace(raw)
}
