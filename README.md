# OpenRouter Analytics

A Python tool suite and CLI for inspecting OpenRouter's 24-hour observed provider analytics: **cache hit rates**, **latency**, **throughput (TPS)**, **uptime**, **effective pricing**, and **token volumes**.

## Features

- **24-Hour Cache Hit Rates**: Query real observed prompt-cache hit rates for any model across all serving providers.
- **Latency & Throughput (TPS)**: Median (p50) and p90 end-to-end latency and tokens-per-second throughput metrics.
- **Uptime Reliability**: 24h trailing uptime percentage for each endpoint.
- **Effective Pricing**: View actual observed input and output prices per million tokens (reflecting cache discounts).
- **Smart Model Resolver**: Automatic typo correction and fuzzy matching (e.g., `z.ai/glm-5.3-flsh` $\rightarrow$ `z-ai/glm-5.3-flash` / `z-ai/glm-5.3-flash-20260826`).
- **Rich Terminal CLI**: Beautiful colored tables with sorting by cache, latency, TPS, uptime, price, or volume.
- **Python Library**: Clean, typed dataclasses (`ModelStats`, `ProviderStats`) with simple helper functions.
- **No API Key Required**: Queries public OpenRouter statistics out-of-the-box. (Optional Management Key supported for private account analytics).

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

### 1. View All Providers for a Model

```bash
openrouter-analytics stats z-ai/glm-5.3-flash
```

*(You can also use shorthand, e.g. `openrouter-analytics stats glm-5.3-flash` or `openrouter-analytics stats z.ai/glm-5.3-flsh`)*

#### Sorting & Filtering Options:
- `--sort [cache|latency|tps|uptime|input|output|tokens|share|name]`: Sort table by specific metric (default: `cache`).
- `--top N`: Limit to the top N providers.
- `--provider <name>`: Filter to a single provider (e.g. `--provider deepinfra`).
- `--json`: Output raw JSON data.

```bash
# Top 5 lowest latency providers
openrouter-analytics stats z-ai/glm-5.3-flash --sort latency --top 5

# Top 5 highest throughput (TPS) providers
openrouter-analytics stats z-ai/glm-5.3-flash --sort tps --top 5

# Highest uptime providers
openrouter-analytics stats z-ai/glm-5.3-flash --sort uptime --top 5
```

---

### 2. Fast Provider Performance Lookup

```bash
# Inspect a specific provider
openrouter-analytics cache z-ai/glm-5.3-flash deepinfra

# Output:
# Model: Z.ai: GLM 5.3 Flash (z-ai/glm-5.3-flash)
# Provider: DeepInfra (deepinfra)
# 24h Cache Hit Rate: 60.9%
# Latency (p50): 1.27s
# Throughput (p50): 30 tps
# Uptime (24h): 99.3%
# Effective Input Price: $0.0453
# Effective Output Price: $0.2881
# Tokens Served: 6.8B (0.7% share)
```

```bash
# Ranked summary of all providers
openrouter-analytics cache z-ai/glm-5.3-flash
```

---

### 3. Compare Specific Providers Side-by-Side

```bash
openrouter-analytics compare z-ai/glm-5.3-flash deepinfra siliconflow novita parasail
```

---

### 4. Search Models

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
print(f"Average Latency: {stats.formatted_avg_latency}")
print(f"Average TPS: {stats.formatted_avg_tps}")
print(f"Total Providers: {stats.provider_count}")

# 2. Iterate providers sorted by latency
for p in stats.sort_by("latency"):
    print(
        f"{p.name:20} "
        f"Latency: {p.formatted_latency:>6} "
        f"TPS: {p.formatted_tps:>7} "
        f"Uptime: {p.formatted_uptime:>6} "
        f"Cache: {p.formatted_cache_hit_rate:>6}"
    )

# 3. Target a specific provider
deepinfra = stats.get_provider("deepinfra")
if deepinfra:
    print(f"DeepInfra Cache Hit Rate: {deepinfra.cache_hit_rate_pct:.1f}%")
    print(f"DeepInfra Latency (p50): {deepinfra.formatted_latency}")
    print(f"DeepInfra Throughput: {deepinfra.formatted_tps}")
    print(f"DeepInfra Uptime (1d): {deepinfra.formatted_uptime}")

# 4. Quick one-line lookup
rate = get_cache_hit_rate("meta-llama/llama-3.1-8b-instruct", "coreweave")
print(f"CoreWeave Cache Hit Rate: {rate * 100:.1f}%")
```

---

## Data Source & Mechanics

- **Cache hit rate**: The proportion of prompt tokens served from provider KV caches today (UTC).
- **Latency (p50)**: Median time-to-first-token or request completion in milliseconds / seconds.
- **Throughput (p50)**: Median tokens-per-second (TPS) generated during the observation window.
- **Uptime (24h)**: Trailing 24-hour endpoint availability percentage.
- **Effective input/output price**: The actual blended price paid per million tokens after caching discounts.
