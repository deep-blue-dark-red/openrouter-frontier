# OpenRouter Analytics

A Python tool suite and CLI for inspecting OpenRouter's 24-hour observed provider analytics: **cache hit rates**, **latency**, **throughput (TPS)**, **uptime**, **effective pricing**, and an **economic ProviderUtility scoring model** for cost and latency routing.

## Features

- **ProviderUtility Scoring Model**: Expected cost per turn factoring in prompt cache hit rates ($h$), Bayesian shrinkage ($h_{used}$), miss/write penalties, time value ($/hr), and uptime failure risk.
- **24-Hour Cache Hit Rates**: Query real observed prompt-cache hit rates for any model across all serving providers.
- **Latency & Throughput (TPS)**: Median (p50) and p90 end-to-end latency and tokens-per-second throughput metrics.
- **Uptime Reliability**: 24h trailing uptime percentage for each endpoint.
- **Effective Pricing**: View actual observed input and output prices per million tokens.
- **Smart Model Resolver**: Automatic typo correction and fuzzy matching (e.g., `z.ai/glm-5.3-flsh` $\rightarrow$ `z-ai/glm-5.3-flash`).
- **Rich Terminal CLI**: Beautiful colored tables with sorting by score/cost, cache, latency, TPS, uptime, price, or volume.
- **Python Library**: Typed dataclasses (`ModelStats`, `ProviderStats`, `ScoreBreakdown`, `ScoringConfig`) with helper functions.

---

## Installation

```bash
cd ~/git/openrouter-analytics
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## ProviderUtility Scoring Model

The scoring model evaluates the expected cost per turn across endpoints:

### Token Cost Term
$$\text{tokenCost} = \frac{C \cdot \left( h_{used} \cdot \text{hitPrice} + (1 - h_{used}) \cdot \text{missPrice} \right) + O \cdot \text{out}}{1{,}000{,}000} + \text{requestFee}$$

- **Inputs** (USD per million tokens from endpoints API):
  - $\text{in} = \text{pricing.prompt}$
  - $\text{out} = \text{pricing.completion}$
  - $\text{read} = \text{pricing.input\_cache\_read}$ (if absent, endpoint has no cache)
  - $\text{write} = \text{pricing.input\_cache\_write}$ (if absent, defaults to $\text{in}$)
- **Derived Prices**:
  - $\text{hitPrice} = \text{read} \text{ if present else } \text{in}$
  - $\text{missPrice} = \text{in} \text{ if read absent else } (\text{write} \text{ if present else } \text{in})$
  - $h = 0 \text{ if read absent else } \text{clamp}(\text{cacheHitRate}, 0, 1)$
- **Bayesian Shrinkage**:
  $$h_{used} = \frac{h \cdot T + \text{prior} \cdot W}{T + W}$$
  where $T$ is the endpoint's observed 24h total tokens, $\text{prior} = 0.5$ (default), and $W = 10^9$ tokens ($1\text{B}$ tokens, default).

### Utility Terms (Time Value & Failure Risk)
$$\text{timeCost} = \left(\frac{\text{TimeValueUsdPerHour}}{3600}\right) \cdot \left(\text{ttft} + \frac{O}{\text{throughput}}\right)$$

$$\text{failureCost} = (1 - \text{uptime}) \cdot \left[ \frac{C \cdot h_{used} \cdot (\text{missPrice} - \text{hitPrice})}{1{,}000{,}000} + \left(\frac{\text{TimeValueUsdPerHour}}{3600}\right) \cdot \text{ttft} \right]$$

$$\text{totalCost} = \text{tokenCost} + \text{timeCost} + \text{failureCost}$$

*Setting `TimeValueUsdPerHour = 0` and `PriceFailures = false` evaluates the pure token cost model.*

---

## Command-Line Interface (CLI)

### 1. Evaluate and Rank Providers by Utility / Cost

```bash
# Pure token cost ranking (2000 prompt tokens, 500 completion tokens)
openrouter-analytics score z-ai/glm-5.3-flash

# Full utility model with $30/hr time value and failure risk
openrouter-analytics score z-ai/glm-5.3-flash --time-value 30 --prompt-tokens 3000 --completion-tokens 800

# Inspect a specific provider breakdown
openrouter-analytics score z-ai/glm-5.3-flash --provider deepinfra --json
```

### 2. View 24h Provider Analytics (Cache, Latency, TPS, Uptime)

```bash
# Sort by highest cache hit rate
openrouter-analytics stats z-ai/glm-5.3-flash --sort cache --top 5

# Sort by lowest total turn cost (score)
openrouter-analytics stats z-ai/glm-5.3-flash --sort score --top 5

# Sort by lowest latency (p50)
openrouter-analytics stats z-ai/glm-5.3-flash --sort latency --top 5
```

### 3. Provider Fast Lookup

```bash
openrouter-analytics cache z-ai/glm-5.3-flash deepinfra
```

### 4. Side-by-Side Provider Comparison

```bash
openrouter-analytics compare z-ai/glm-5.3-flash deepinfra siliconflow novita
```

---

## Python API Usage

```python
from openrouter_analytics import (
    get_model_stats,
    score_model_providers,
    ScoringConfig,
)

# 1. Evaluate and rank all providers using the scoring model
cfg = ScoringConfig(
    prompt_tokens=2000,
    completion_tokens=500,
    time_value_usd_per_hour=30.0,  # $30/hr time value
    price_failures=True,           # factor in uptime risk
    prior=0.5,                     # shrinkage prior
    prior_weight_tokens=1e9,       # shrinkage weight
)

ranked_scores = score_model_providers("z.ai/glm-5.3-flsh", config=cfg)

for s in ranked_scores[:5]:
    print(f"#{s.rank:2d} {s.provider_name:15} | Total Cost: {s.formatted_total_cost} "
          f"(Token: {s.formatted_token_cost}, Time: {s.formatted_time_cost}) | "
          f"h_used: {s.formatted_h_used} (raw: {s.formatted_h_raw})")

# 2. Evaluate a single provider
stats = get_model_stats("z.ai/glm-5.3-flsh")
deepinfra = stats.get_provider("deepinfra")
score = deepinfra.evaluate_score(cfg)

print(f"DeepInfra Token Cost:   {score.formatted_token_cost}")
print(f"DeepInfra Failure Risk: {score.formatted_failure_cost}")
print(f"DeepInfra Total Cost:   {score.formatted_total_cost}")
```
