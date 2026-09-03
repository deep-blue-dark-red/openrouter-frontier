# OpenRouter Analytics

A high-performance Python tool suite and CLI for inspecting OpenRouter's 24-hour observed provider metrics: **cache hit rates**, **latency (TTFT)**, **throughput (TPS)**, **uptime reliability**, **effective pricing**, the **ProviderUtility economic scoring model**, and **Pareto efficiency frontier analysis**.

---

## Key Capabilities

- **ProviderUtility Scoring Model**: Calculates expected cost per turn factoring in prompt cache hit rates ($h$), Bayesian shrinkage ($h_{used}$), miss/write penalties, time value ($/hr), and uptime failure risk.
- **Pareto Frontier Analysis (`frontier.py`)**: Multi-objective trade-off analysis balancing Cost vs. Latency vs. Throughput vs. Cache Hit Rate vs. Uptime. Identifies non-dominated Pareto-optimal providers.
- **Standalone Fast Scripts**: Dedicated executable scripts (`./score_providers.py` and `./frontier.py`) with zero boilerplate.
- **Primary Quantization Matching**: Automatically filters to the official primary variant (e.g. `FP8`), matching the OpenRouter web pricing page by default.
- **Sub-Second Performance**: Optimized with forced IPv4 resolution (eliminating the 10-second macOS IPv6 connection stall), HTTP Keep-Alive session pooling, and multi-tier 5-minute disk caching.
  - **Warm execution**: `~0.11s` (110ms)
  - **Cold execution**: `< 1.0s`
- **Smart Model Resolver**: Automatic typo correction and fuzzy matching (e.g., `z.ai/glm-5.3-flash` $\rightarrow$ `z-ai/glm-5.3-flash-20260826`).
- **Clean Modern Terminal Tables**: Rendered with modern connected dashes (`─`), space separation, and no vertical bars.

---

## Installation

```bash
cd ~/git/openrouter-analytics
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## Executable Scripts

### 1. `score_providers.py` — Cost & Utility Scoring

Evaluates providers and ranks them by expected cost per turn.

```bash
# Pure token cost model (Z.ai ranks #1)
./score_providers.py z.ai/glm-5.3-flash --top 5

# Full utility model with $30/hr time opportunity cost
./score_providers.py z.ai/glm-5.3-flash --time-value 30 --top 5

# Include unquantized community variants
./score_providers.py z.ai/glm-5.3-flash --all-quants

# Output raw JSON
./score_providers.py z.ai/glm-5.3-flash --json
```

#### Example Output:

```text
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ProviderUtility Evaluation: Z.ai: GLM 5.3 Flash (z-ai/glm-5.3-flash)
Mode: Full Utility Model  •  Turn: 2000 prompt + 500 completion tokens  •  Time Value: $0.00/hr
Shrinkage: prior=50%, weight=1.0B tokens  •  Discounts: Applied  •  Failure Risk: Yes
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Provider           Scored Cost   Token Cost   Fail Risk   h(used)   h(pub)   Hit $/M   Miss $/M   Latency    TPS   Uptime
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Z.ai                 $0.000086    $0.000085   $0.000002     87.9%    88.0%   $0.0075    $0.0375     4.68s     26    96.7%
NovitaAI             $0.000091    $0.000090   $0.000001     79.0%    79.6%   $0.0075    $0.0375     1.59s     39    98.9%
GMICloud             $0.000099    $0.000096   $0.000003     68.8%    69.0%   $0.0075    $0.0375     5.75s     18    93.5%
DeepInfra            $0.000102    $0.000102   $0.000000     59.0%    60.1%   $0.0075    $0.0375     0.87s     32    99.4%
Modal                $0.000123    $0.000123   $0.000000     74.8%    75.8%   $0.0100    $0.0500     0.55s     45    99.9%
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Lower Total Cost represents higher utility. Ranks include cache hit rates, shrinkage, and endpoint metrics.
```

---

### 2. `frontier.py` — Pareto Frontier Efficiency

Instead of picking an arbitrary time-value dollar scalar, `frontier.py` finds which providers form the non-dominated Pareto frontier across Cost, Latency, Throughput, Cache Hit Rate, and Uptime.

```bash
# View all candidates and frontier classification
./frontier.py z.ai/glm-5.3-flash

# View ONLY Pareto-optimal providers
./frontier.py z.ai/glm-5.3-flash --optimal-only
```

#### Example Output:

```text
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Pareto Frontier Evaluation: Z.ai: GLM 5.3 Flash (z-ai/glm-5.3-flash)
Multi-Objective Pareto Analysis  •  Turn: 2000 prompt + 500 completion tokens
Evaluation: Cost vs Latency vs TPS vs Cache vs Uptime  •  Quantization: FP8
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Provider           Scored Cost   Token Cost   Latency    TPS   h(used)   Uptime  Pareto Frontier  Niche / Advantage               
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Z.ai                 $0.000086    $0.000085     4.68s     26     87.9%    96.7%  ★ OPTIMAL        Lowest Cost • Best Cache Hit    
NovitaAI             $0.000091    $0.000090     1.59s     39     79.0%    98.9%  ★ OPTIMAL        Balanced Trade-off              
GMICloud             $0.000099    $0.000096     5.75s     18     68.8%    93.5%  Dominated        --                              
DeepInfra            $0.000102    $0.000102     0.87s     32     59.0%    99.4%  ★ OPTIMAL        Balanced Trade-off              
Modal                $0.000123    $0.000123     0.55s     45     74.8%    99.9%  ★ OPTIMAL        Lowest Latency • Highest Uptime 
Morph                $0.000340    $0.000336     1.16s     22     67.5%    97.9%  Dominated        --                              
SiliconFlow          $0.000345    $0.000340     2.13s     29     87.4%    97.6%  ★ OPTIMAL        Best Cache Hit                  
NextBit              $0.000355    $0.000350     2.56s     38     83.4%    97.3%  ★ OPTIMAL        Balanced Trade-off              
Parasail             $0.000366    $0.000358     1.24s     68     79.9%    95.7%  ★ OPTIMAL        Balanced Trade-off              
Phala                $0.000387    $0.000383     1.58s     38     69.4%    98.1%  Dominated        --                              
Baseten              $0.000425    $0.000425     1.32s     71     52.1%    99.6%  ★ OPTIMAL        Highest TPS                     
Sail Research        $0.000445    $0.000444     2.30s     37     44.0%    99.5%  Dominated        --                              
Reka AI              $0.000451    $0.000450     1.75s     51     41.5%    99.9%  ★ OPTIMAL        Highest Uptime                  
io.net               $0.000459    $0.000445     1.76s     11     43.8%    86.8%  Dominated        --                              
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
★ OPTIMAL indicates non-dominated Pareto-optimal providers on the cost-performance efficiency frontier.
```

---

## ProviderUtility Scoring Mathematical Model

### 1. Token Cost Term
$$\text{tokenCost} = \frac{C \cdot \left( h_{used} \cdot \text{hitPrice} + (1 - h_{used}) \cdot \text{missPrice} \right) + O \cdot \text{out}}{1{,}000{,}000} + \text{requestFee}$$

- **Inputs** (USD per million tokens, scaled from endpoints API):
  - $\text{in} = \text{pricing.prompt}$ (input price)
  - $\text{out} = \text{pricing.completion}$ (output price)
  - $\text{read} = \text{pricing.input\_cache\_read}$ (absent if caching unsupported)
  - $\text{write} = \text{pricing.input\_cache\_write}$ (absent if billed as input)
- **Derived Prices**:
  - $\text{hitPrice} = \text{read} \text{ if present else } \text{in}$
  - $\text{missPrice} = \text{in} \text{ if read absent else } (\text{write} \text{ if present else } \text{in})$
  - $h = 0 \text{ if read absent else } \text{clamp}(\text{cacheHitRate}, 0, 1)$
- **Bayesian Shrinkage**:
  $$h_{used} = \frac{h \cdot T + \text{prior} \cdot W}{T + W}$$
  where $T$ is the endpoint's observed 24h total tokens, $\text{prior} = 0.5$ (default), and $W = 10^9$ tokens ($1\text{B}$ tokens, default).

### 2. Utility Terms (Time Opportunity Cost & Failure Risk)
$$\text{timeCost} = \left(\frac{\text{TimeValueUsdPerHour}}{3600}\right) \cdot \left(\text{ttft} + \frac{O}{\text{throughput}}\right)$$

$$\text{failureCost} = (1 - \text{uptime}) \cdot \left[ \frac{C \cdot h_{used} \cdot (\text{missPrice} - \text{hitPrice})}{1{,}000{,}000} + \left(\frac{\text{TimeValueUsdPerHour}}{3600}\right) \cdot \text{ttft} \right]$$

$$\text{totalCost} = \text{tokenCost} + \text{timeCost} + \text{failureCost}$$

*Setting `TimeValueUsdPerHour = 0` and `PriceFailures = false` evaluates the pure token cost model.*

---

## Python API Usage

```python
from openrouter_analytics import (
    score_model_providers,
    get_model_stats,
    ScoringConfig,
)

# 1. Rank providers with pure token cost
scores = score_model_providers("z-ai/glm-5.3-flash")
best = scores[0]
print(f"Top Provider: {best.provider_name} -> {best.formatted_token_cost} / turn")

# 2. Custom conversation simulation
cfg = ScoringConfig(
    prompt_tokens=4000,
    completion_tokens=1000,
    time_value_usd_per_hour=0.0,
    price_failures=True,
)
ranked = score_model_providers("z-ai/glm-5.3-flash", config=cfg)
for s in ranked[:3]:
    print(f"{s.provider_name:<15} Scored Cost: {s.formatted_total_cost} | h_used: {s.formatted_h_used}")
```

---

## CLI Tools

```bash
# Rich colored tables with sorting
openrouter-analytics score z-ai/glm-5.3-flash --top 5

# Inspect 24h metrics sorted by cache hit rate
openrouter-analytics stats z-ai/glm-5.3-flash --sort cache --top 5

# Compare specific providers side-by-side
openrouter-analytics compare z-ai/glm-5.3-flash z-ai novita deepinfra
```

---

## License

MIT
