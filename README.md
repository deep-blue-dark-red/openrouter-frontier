# OpenRouter Analytics

A high-performance Python tool suite and CLI for inspecting OpenRouter's 24-hour observed provider metrics: **cache hit rates**, **latency (TTFT)**, **throughput (TPS)**, **uptime reliability**, **effective pricing**, the **ProviderUtility economic scoring model**, and **Pareto efficiency frontier analysis**.

---

## Key Capabilities

- **ProviderUtility Scoring Model**: Calculates expected cost per turn factoring in prompt cache hit rates ($h$), Bayesian shrinkage ($h_{used}$), miss/write penalties, time value ($/hr), and uptime failure risk.
- **Pareto Frontier Analysis (`provider_frontier.py`)**: Multi-objective trade-off analysis balancing Cost vs. Latency vs. Throughput vs. Cache Hit Rate vs. Uptime. Identifies non-dominated Pareto-optimal providers.
- **Standalone Fast Scripts**: Dedicated executable scripts (`./score_providers.py` and `./provider_frontier.py`) with zero boilerplate.
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

### 1. `openrouter-tui` — Interactive Terminal Explorer

Interactive terminal UI (inspired by `makiss`) with:
- **Model List View**: Models sorted by Pareto frontier ranking (top items on the frontier starting from lowest cost to highest, followed by distance to frontier).
- **Live Fuzzy Search**: Punctuation-agnostic search (`zai` matches `z-ai` and `z.ai`, `claude37` matches `claude-3.7-sonnet`, `gemini25` matches `gemini-2.5-flash`).
- **Provider View**: Press **Enter** on any model to view its serving endpoints sorted by ProviderUtility Scored Cost.
- **Spec Card**: Press **Enter** on a provider to inspect detailed pricing, latency percentiles, and Bayesian shrinkage metrics.
- **Controls**: Up/Down / Ctrl-P/N / Mouse scroll to navigate, PgUp/PgDn to jump 5, Tab/a to toggle quantization filters, Esc / Backspace to return or exit.

```bash
# Launch interactive TUI from any directory
openrouter-tui

# Or run locally from repo
./openrouter-tui
```

---
### 2. `score_providers.py` — Cost & Utility Scoring

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
Provider           Scored Cost   Token Cost   Fail Risk   CacheHit   h(pub)   Hit $/M   Miss $/M   Latency    TPS   Uptime
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

### 3. `provider_frontier.py` — Pareto Frontier Efficiency

Instead of picking an arbitrary time-value dollar scalar, `provider_frontier.py` finds which providers form the non-dominated Pareto frontier across Cost, Latency, Throughput, Cache Hit Rate, and Uptime.

```bash
# View all candidates and frontier classification
./provider_frontier.py z.ai/glm-5.3-flash

# View ONLY Pareto-optimal providers
./provider_frontier.py z.ai/glm-5.3-flash --optimal-only
```

#### Example Output:

```text
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Pareto Frontier Evaluation: Z.ai: GLM 5.3 Flash (z-ai/glm-5.3-flash)
Multi-Objective Pareto Analysis  •  Turn: 2000 prompt + 500 completion tokens
Evaluation: Cost vs Latency vs TPS vs Cache vs Uptime  •  Quantization: FP8
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Provider           Scored Cost   Token Cost   Latency    TPS   CacheHit   Uptime  Pareto Frontier  Niche / Advantage               
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

### 3. `model_provider_frontier.py` — Catalog-Wide Model Pareto Frontier

Computes the multi-objective Pareto efficiency frontier across all **400+ models** in OpenRouter's catalog, finding optimal trade-offs between **Turn Cost**, **Context Window Size**, and **Cache Read Pricing**.

```bash
# View Pareto-optimal models across the entire catalog
./model_provider_frontier.py --optimal-only

# Filter to a specific model family (e.g. Flash, Claude, Qwen)
./model_provider_frontier.py -q flash --optimal-only

# Output raw JSON
./model_provider_frontier.py --json
```

#### Example Output:

```text
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
OpenRouter Catalog-Wide Model Pareto Frontier
Multi-Objective Pareto Analysis  •  Turn: 2000 prompt + 500 completion tokens
Evaluation: Turn Cost vs Context Length vs Cache Read Pricing  •  14 Pareto-Optimal Models
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Model ID                              Turn Cost  Prompt $/M   Compl $/M   Read $/M     Context  Frontier      Niche / Advantage           
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
mistralai/mistral-nemo                $0.000053     $0.0190     $0.0300         --        131k  ★ OPTIMAL     Cheapest Model              
inclusionai/ling-3.0-flash            $0.000073     $0.0210     $0.0630    $0.0042        262k  ★ OPTIMAL     Cost/Context Trade-off      
nex-agi/nex-n2-mini                   $0.000100     $0.0250     $0.1000    $0.0025        262k  ★ OPTIMAL     Cost/Context Trade-off      
upstage/solar-pro4                    $0.000120     $0.0300     $0.1200    $0.0060        524k  ★ OPTIMAL     Cost/Context Trade-off      
qwen/qwen3.7-flash                    $0.000125     $0.0300     $0.1300    $0.0060       1000k  ★ OPTIMAL     1M Context                  
openai/gpt-5-nano:batch               $0.000150     $0.0250     $0.2000    $0.0025        400k  ★ OPTIMAL     Cost/Context Trade-off      
~deepseek/deepseek-v4-flash-latest    $0.000180     $0.0500     $0.1600    $0.0130       1310k  ★ OPTIMAL     1M Context                  
google/gemini-2.5-flash-lite:batch    $0.000200     $0.0500     $0.2000    $0.0100       1048k  ★ OPTIMAL     1M Context                  
poolside/laguna-s-2.1                 $0.000270     $0.0900     $0.1800    $0.0090       1048k  ★ OPTIMAL     1M Context                  
meta/muse-spark-1.3-contributor       $0.000300     $0.1000     $0.2000    $0.0020       1048k  ★ OPTIMAL     1M Context • Cheapest Cache Read
x-ai/grok-4.20                        $0.003750     $1.2500     $2.5000    $0.2000       2000k  ★ OPTIMAL     Max Context (2000k)         
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

### 5. `get_models.py` — Model Catalog Query & Inspector

Fast search and inspection for all 400+ models on OpenRouter with prompt caching, context length, and pricing breakdown.

```bash
# Inspect full specifications and pricing for a single model
./get_models.py z-ai/glm-5.3-flash

# Search by keyword
./get_models.py -q flash --top 10

# Filter by creator and sort by context window
./get_models.py --creator anthropic --sort context

# Filter only models supporting prompt caching
./get_models.py --caching --sort price --top 10

# Force refresh from OpenRouter API
./get_models.py --refresh
```

#### Example Output:

```text
────────────────────────────────────────────────────────────────────────────────
Model: Z.ai: GLM 5.3 Flash
ID:    z-ai/glm-5.3-flash
Slug:  z-ai/glm-5.3-flash-20260826
────────────────────────────────────────────────────────────────────────────────
  Context Window:        1,310,720 tokens
  Max Output Tokens:     131,072
  Modality:              text+image+video->text

  Pricing (USD per million tokens):
    Prompt (Input):      $0.0750 / M
    Completion (Output): $0.2500 / M
    Cache Read:          $0.0150 / M
    Cache Write:         -- / M

  Description:
    GLM-5.3-Flash is a native multimodal model from Z.ai. It is suited for efficient coding an
────────────────────────────────────────────────────────────────────────────────
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
