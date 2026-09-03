# OpenRouter Analytics

A Python tool suite and CLI for inspecting OpenRouter's 24-hour observed provider analytics, cache hit rates, effective pricing, and token volumes.

## Features

- **24-Hour Cache Hit Rates**: Query real observed prompt-cache hit rates for any model across all serving providers.
- **Effective Pricing**: View actual observed input and output prices per million tokens (after caching discounts).
- **Smart Model Resolver**: Automatic typo correction and fuzzy matching (e.g., `z.ai/glm-5.3-flsh` seamlessly maps to `z-ai/glm-5.3-flash` and its canonical permaslug `z-ai/glm-5.3-flash-20260826`).
- **Rich Terminal CLI**: Beautiful colored tables showing rankings, token share, and weighted averages.
- **Python Library**: Clean, typed dataclasses (`ModelStats`, `ProviderStats`) and simple helper functions.
- **No API Key Required**: Queries public OpenRouter frontend statistics out-of-the-box. (Optional Management Key supported for private account analytics).

---

## Installation

```bash
cd ~/git/openrouter-analytics
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## Command-Line Interface (CLI)

### 1. View all providers for a model

```bash
openrouter-analytics stats z-ai/glm-5.3-flash
```

*(You can also use shorthand, e.g. `openrouter-analytics stats glm-5.3-flash` or `openrouter-analytics stats z.ai/glm-5.3-flsh`)*

#### Options:
- `--sort [cache|input|output|tokens|share|name]`: Sort table by specific metric (default: `cache`).
- `--top N`: Limit to the top N providers.
- `--provider <name>`: Filter to a single provider (e.g. `--provider deepinfra`).
- `--json`: Output raw JSON data.

```bash
# Top 5 providers for GLM 5.3 Flash by cache hit rate
openrouter-analytics stats z-ai/glm-5.3-flash --sort cache --top 5

# JSON output
openrouter-analytics stats z-ai/glm-5.3-flash --provider deepinfra --json
```

---

### 2. Fast cache hit rate lookup

```bash
# Get the specific rate for a single provider
openrouter-analytics cache z-ai/glm-5.3-flash deepinfra

# Output:
# Model: Z.ai: GLM 5.3 Flash (z-ai/glm-5.3-flash)
# Provider: DeepInfra (deepinfra)
# 24h Cache Hit Rate: 60.9%
# Effective Input Price: $0.0453 /M
# Effective Output Price: $0.2881 /M
# Tokens Served: 6.8B (1.5% share)
```

```bash
# List all providers ranked by cache hit rate
openrouter-analytics cache z-ai/glm-5.3-flash
```

---

### 3. Compare specific providers side-by-side

```bash
openrouter-analytics compare z-ai/glm-5.3-flash deepinfra siliconflow parasail novita
```

---

### 4. Search models

```bash
openrouter-analytics search glm-5
```

---

## Python API Usage

```python
from openrouter_analytics import get_model_stats, get_cache_hit_rate

# 1. Fetch full stats for a model
stats = get_model_stats("z.ai/glm-5.3-flsh")

print(f"Model: {stats.model_name}")
print(f"Weighted 24h Cache Hit Rate: {stats.formatted_weighted_cache_hit_rate}")
print(f"Total Providers: {stats.provider_count}")

# 2. Iterate providers sorted by cache hit rate
for provider in stats.sort_by("cache"):
    print(
        f"{provider.name:20} "
        f"Cache: {provider.formatted_cache_hit_rate:>7} "
        f"Input: {provider.formatted_input_price:>12} "
        f"Tokens: {provider.formatted_tokens:>6}"
    )

# 3. Target a specific provider
deepinfra = stats.get_provider("deepinfra")
if deepinfra:
    print(f"DeepInfra Cache Hit Rate: {deepinfra.cache_hit_rate_pct:.1f}%")
    print(f"DeepInfra Effective Input: {deepinfra.formatted_input_price}")

# 4. Quick one-line lookup
rate = get_cache_hit_rate("meta-llama/llama-3.1-8b-instruct", "coreweave")
print(f"CoreWeave Llama 3.1 8B Cache Hit Rate: {rate * 100:.1f}%")
```

---

## Data Source & Mechanics

OpenRouter computes these statistics across observed traffic **today (UTC) / trailing 24 hours**:
- **Cache hit rate**: The proportion of prompt tokens that successfully hit provider KV caches.
- **Effective input price**: The blended average price paid per million input tokens, factoring in prompt caching discounts (which often offer 50–90% savings).
