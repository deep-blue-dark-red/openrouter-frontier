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

### 4. `model_frontier.py` — Benchmark-Driven Model Pareto Frontier

Computes the Cost vs. Quality Pareto efficiency frontier using live Artificial Analysis quality benchmark scores against OpenRouter catalog pricing. Detects non-dominated models, calculates normalized marginal efficiency, and highlights the **Knee Point** (maximum quality gain per dollar).

```bash
# View Intelligence Pareto Frontier with Knee point
./model_frontier.py

# Evaluate Coding Benchmark Index
./model_frontier.py --metric coding

# Evaluate Agentic Index using traffic-weighted pricing
./model_frontier.py --metric agentic --price-source weighted

# Output full JSON
./model_frontier.py --format json
```

#### Example Output:

```text
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
OpenRouter Pareto Frontier [INTELLIGENCE] (8 non-dominated of 87 models)
Price metric: LIST ($/1M prompt)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
MODEL                                                     ID                                Intelligence Score  Cost ($/1M prompt)  STATUS        
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Ling 3.0 Flash                                            inclusionai/ling-3.0-flash                37.8             $0.02  ON FRONTIER   
Solar Pro 4                                               upstage/solar-pro4                        41.6             $0.03  ON FRONTIER   
GLM-5.3-Flash                                             z-ai/glm-5.3-flash                        57.5             $0.07  ← KNEE        
Gemini 3.8 Flash (high)                                   google/gemini-3.8-flash                   58.7             $0.75  ON FRONTIER   
GLM-5.3 (max)                                             z-ai/glm-5.3                              59.5             $1.40  ON FRONTIER   
GPT-5.6 Sol (max)                                         openai/gpt-5.6-sol                        60.9             $2.00  ON FRONTIER   
Claude Opus 5 (Adaptive Reasoning, Max Effort)            anthropic/claude-opus-5                   63.1             $5.00  ON FRONTIER   
Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Defaul  anthropic/claude-fable-5.1                65.7            $10.00  ON FRONTIER   
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
← KNEE indicates the optimal trade-off point with maximum quality score gain per dollar.
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

## Pareto Frontiers

This section provides the rigorous mathematical formulation, objective spaces, Bayesian models, and algorithms used to compute both the **Model Pareto Frontier** (Intelligence/Coding Quality vs. Token Cost) and the **Provider Pareto Frontier & Provider Utility Scorer** (Turn Cost, Latency, TPS, Cache Hit Rate, and Reliability).

---

### 1. Model Pareto Frontier: Quality vs. Cost

The model Pareto frontier identifies non-dominated LLMs balancing empirical benchmark intelligence/coding capability against token pricing.

#### 1.1 Objective Dimensions
Given a set of candidate models $\mathcal{M} = \{m_1, m_2, \dots, m_N\}$, each model is mapped to a two-dimensional performance coordinate:

1. **Cost Metric $c(m)$ (Minimize $\downarrow$)**:
   - **Catalog List Price**: $c(m) = P_{\text{prompt}}(m)$ in USD per 1M input tokens.
   - **Traffic-Weighted Effective Price**: $c(m) = P_{\text{weighted}}(m)$ derived from production routing volume.
   - **Standard Turn Price**: $c(m) = \frac{C \cdot P_{\text{prompt}} + O \cdot P_{\text{completion}}}{1{,}000{,}000}$ for a standardized turn of $C$ prompt and $O$ completion tokens.

2. **Quality Benchmark Index $s(m)$ (Maximize $\uparrow$)**:
   Sourced from OpenRouter's live Artificial Analysis benchmark rankings (`/api/frontend/v1/rankings/benchmarks`):
   - **Intelligence Index** $\in [0, 100]$: General reasoning, knowledge retrieval, and math/logic synthesis.
   - **Coding Index** $\in [0, 100]$: Code generation, syntax instruction following, and debugging accuracy.
   - **Agentic Index** $\in [0, 100]$: Tool calling, multi-step environment planning, and goal execution.

#### 1.2 Mathematical Formulation of Pareto Dominance
A candidate model $A$ **strictly dominates** candidate model $B$ ($A \succ B$) if and only if $A$ achieves equal or lower cost while achieving equal or higher quality, with at least one strict inequality:
$$A \succ B \iff \big(c(A) \le c(B) \land s(A) \ge s(B)\big) \land \big(c(A) < c(B) \lor s(A) > s(B)\big)$$

The **Pareto Frontier** $\mathcal{F}$ is the non-dominated subset of $\mathcal{M}$:
$$\mathcal{F} = \{m \in \mathcal{M} \mid \nexists \, m' \in \mathcal{M} \text{ such that } m' \succ m\}$$

#### 1.3 Upper-Hull Monotonic Sweep Algorithm
The frontier is extracted in $O(N \log N)$ time:
1. Filter out invalid candidates where $c_i \le 0$ or $s_i \le 0$.
2. Sort candidates by cost ascending; break cost ties by score descending:
   $$\mathcal{M}_{\text{sorted}} = \text{sort}\Big(\mathcal{M}, \ \text{key} = \big(c(m), -s(m)\big)\Big)$$
3. Perform a linear sweep tracking the running maximum score $s_{\max}$:
   - Initialize $s_{\max} \leftarrow -\infty$, $\mathcal{F} \leftarrow []$.
   - For each model $m \in \mathcal{M}_{\text{sorted}}$:
     $$\text{If } s(m) > s_{\max}: \quad \mathcal{F} \leftarrow \mathcal{F} \cup \{m\}, \quad s_{\max} \leftarrow s(m)$$
   Because cost is monotonically non-decreasing, any candidate with a quality score strictly greater than all preceding cheaper models is guaranteed to be non-dominated.

#### 1.4 Knee Point Detection (Maximum Marginal Efficiency)
The **Knee Point** identifies the model on the frontier providing the maximum normalized quality gain per dollar before diminishing returns set in:

1. Identify the boundary endpoints of the frontier:
   $$\text{Cheapest Anchor: } (c_{\min}, s_{\min}) = \mathcal{F}[0], \qquad \text{Highest Quality Anchor: } (c_{\max}, s_{\max}) = \mathcal{F}[-1]$$

2. Normalize cost and score coordinates into the unit square $[0, 1]^2$:
   $$\tilde{c}_i = \frac{c_i - c_{\min}}{c_{\max} - c_{\min}}, \qquad \tilde{s}_i = \frac{s_i - s_{\min}}{s_{\max} - s_{\min}}$$

3. Compute the perpendicular elevation above the linear chord connecting $(0, 0)$ to $(1, 1)$, which corresponds to the marginal advantage:
   $$\text{Gain}_i = \tilde{s}_i - \tilde{c}_i$$

4. The Knee Point $k^*$ is the frontier model maximizing this marginal gain:
   $$k^* = \arg\max_{i \in \mathcal{F}} (\tilde{s}_i - \tilde{c}_i)$$

*Example*: On the OpenRouter Intelligence frontier, **GLM-5.3-Flash** achieves $\text{Score} = 57.5$ at $c = \$0.075 / \text{1M}$, yielding $\text{Gain} = 0.697$—the global maximum across all 87 models.

#### 1.5 Quality Deficit for Dominated Models
For any dominated model $d \notin \mathcal{F}$, its distance to the frontier measures the intelligence/coding score forfeited compared to the best available alternative at or below its price point:
$$\text{Dist}(d) = \max_{\{f \in \mathcal{F} \mid c(f) \le c(d)\}} s(f) - s(d)$$

Models are ranked in the TUI by:
1. $\text{On Frontier}$ models first (ordered by cost ascending).
2. $\text{Dominated}$ models next (ordered by distance to frontier $\text{Dist}(d)$ ascending).

---

### 2. Provider Scoring Mathematical Model

When a model is chosen, its execution can be routed to multiple independent host endpoints. The **ProviderUtility** framework computes the true economic cost per turn for each provider, accounting for prompt caching economics, Bayesian shrinkage, network latency, token throughput, and service reliability.

#### 2.1 Standard Turn Simulation Parameters
- $C$: Number of prompt (input) tokens (default: 2,000).
- $O$: Number of completion (output) tokens (default: 500).
- $V_{\text{hour}}$: Opportunity value of user/agent time in USD/hour (default: $0.00/hr; e.g., $60.00/hr for engineering agents).
- $V_{\text{sec}} = \frac{V_{\text{hour}}}{3600}$: Time value per second.

#### 2.2 Token Pricing & Effective Cache Rates
For an endpoint with published pricing per million tokens:
- $P_{\text{in}}$: Prompt price ($\text{pricing.prompt}$).
- $P_{\text{out}}$: Completion price ($\text{pricing.completion}$).
- $P_{\text{read}}$: Cache read price ($\text{pricing.input\_cache\_read}$).
- $P_{\text{write}}$: Cache write price ($\text{pricing.input\_cache\_write}$).

Derived operational prices:
$$P_{\text{hit}} = \begin{cases} P_{\text{read}} & \text{if caching supported} \\ P_{\text{in}} & \text{otherwise} \end{cases}$$
$$P_{\text{miss}} = \begin{cases} P_{\text{in}} & \text{if caching unsupported} \\ P_{\text{write}} & \text{if write fee specified} \\ P_{\text{in}} & \text{otherwise} \end{cases}$$

#### 2.3 Bayesian Empirical Shrinkage of Cache Hit Rates
Published 24-hour cache hit rates $h_{\text{raw}}$ can fluctuate or suffer from sparse sample sizes on low-traffic endpoints. To prevent optimistic bias on unproven providers, we apply an empirical Bayes shrinkage estimator toward an uninformative prior:
$$h_{\text{used}} = \frac{h_{\text{raw}} \cdot T + \text{Prior} \cdot W}{T + W}$$
where:
- $T$: Endpoint observed 24h token volume.
- $\text{Prior} = 0.50$ ($50\%$ prior expectation).
- $W = 10^9$ tokens ($1\text{B}$ pseudo-observation weight).

As an endpoint logs billions of tokens ($T \gg W$), $h_{\text{used}} \to h_{\text{raw}}$. For low-volume endpoints ($T \to 0$), $h_{\text{used}}$ safely shrinks toward $50\%$.

#### 2.4 Token Cost Component
$$\text{TokenCost} = \frac{C \cdot \left[ h_{\text{used}} \cdot P_{\text{hit}} + (1 - h_{\text{used}}) \cdot P_{\text{miss}} \right] + O \cdot P_{\text{out}}}{1{,}000{,}000} + \text{Fee}_{\text{request}}$$

#### 2.5 Time Opportunity Cost Component
Total turn turnaround time includes Time to First Token ($\text{TTFT}$) plus streaming duration:
$$\text{Duration} = \text{TTFT} + \frac{O}{\text{Throughput}_{\text{TPS}}}$$
$$\text{TimeCost} = V_{\text{sec}} \cdot \text{Duration} = \left(\frac{V_{\text{hour}}}{3600}\right) \cdot \left(\text{TTFT} + \frac{O}{\text{TPS}}\right)$$

#### 2.6 Failure Risk Penalty Component
When a request fails, the turn must be retried on an alternate endpoint:
1. Cache locality is broken: the prompt tokens must be re-sent as an un-cached cache-miss, forfeiting the cache discount.
2. Latency is doubled: the client paid the $\text{TTFT}$ delay without receiving tokens.

With endpoint availability rate $U = \text{clamp}(\text{uptime}, 0.5, 1.0)$ and failure probability $P_{\text{fail}} = 1 - U$:
$$\text{FailureCost} = (1 - U) \cdot \left[ \frac{C \cdot h_{\text{used}} \cdot (P_{\text{miss}} - P_{\text{hit}})}{1{,}000{,}000} + V_{\text{sec}} \cdot \text{TTFT} \right]$$

#### 2.7 Total Scored Turn Cost
$$\text{ScoredCost} = \text{TokenCost} + \text{TimeCost} + \text{FailureCost}$$
Endpoints are sorted in ascending order of $\text{ScoredCost}$. The rank #1 provider delivers the strictly optimal turn utility.

---

### 3. Multi-Objective Provider Pareto Frontier

Rather than assuming a single time-value parameter $V_{\text{hour}}$, `provider_frontier.py` performs multi-objective Pareto optimization across 5 concurrent operational dimensions.

#### 3.1 Objective Vector
Each provider endpoint $p$ is mapped to a 5-dimensional performance tuple:
$$\mathbf{v}(p) = \big(\text{ScoredCost}(p), \ \text{TTFT}(p), \ -\text{TPS}(p), \ -h_{\text{used}}(p), \ -\text{Uptime}(p)\big)$$
where minimization is applied uniformly across all 5 coordinates (negating objectives that seek maximization).

#### 3.2 Dominance Criterion
Provider $p_1$ dominates provider $p_2$ ($p_1 \succ p_2$) if and only if $p_1$ is at least as good in all 5 dimensions and strictly superior in at least one:
$$\begin{cases}
\text{ScoredCost}(p_1) \le \text{ScoredCost}(p_2) \\
\text{TTFT}(p_1) \le \text{TTFT}(p_2) \\
\text{TPS}(p_1) \ge \text{TPS}(p_2) \\
h_{\text{used}}(p_1) \ge h_{\text{used}}(p_2) \\
\text{Uptime}(p_1) \ge \text{Uptime}(p_2)
\end{cases}$$
with at least one strict inequality.

#### 3.3 Frontier Classification & Niche Advantages
Endpoints in the non-dominated set $\mathcal{P}^*$ are classified as **★ OPTIMAL** and tagged with their specialized operational niche:
- **Lowest Cost**: Achieves $\min_{p} \text{ScoredCost}(p)$.
- **Lowest Latency**: Achieves $\min_{p} \text{TTFT}(p)$.
- **Highest TPS**: Achieves $\max_{p} \text{TPS}(p)$.
- **Best Cache Hit**: Achieves $\max_{p} h_{\text{used}}(p)$.
- **Highest Uptime**: Achieves $\max_{p} \text{Uptime}(p)$.
- **Balanced Trade-off**: Non-dominated generalist offering strong multi-criteria compromise.

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
