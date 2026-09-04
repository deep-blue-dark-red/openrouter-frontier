# OpenRouter Frontier

**The price of an LLM call is not the list price. It is the expected cost of the whole task, and
that depends on who serves it.** Two providers selling the same model at the same price can
differ by 25% in what you actually pay, because one hits the prompt cache 88% of the time and
the other under 50%, one drops one request in ten and throws away your cached prefix, one makes
you wait four seconds for the first token. Nobody prices that. This does.

OpenRouter Frontier pulls the observed 24-hour metrics for every endpoint serving a model —
prompt **cache hit rate**, p50 **latency** and **throughput**, **uptime**, effective and list
**pricing** — and turns them into the expected cost of a whole multi-turn task, then reasons about it:

- **ProviderScore** – the expected dollars a whole task costs on each endpoint. A task is $N$
  turns whose transcript is resubmitted every turn, so the growing prefix is paid $N$ times:
  cache economics from the endpoint's observed 24-hour hit rate, the value of your time waiting
  on its latency, throughput, and the re-prefill of the prefix after a cache miss, and what a
  failure costs when the task is retried cold on a fallback. The default profile is a task whose
  context grows to 300k tokens with 10k tokens of output, time valued at $20/hr. The full model
  is in [Scoring model](#scoring-model) and derived in
  [docs/task_cost_model.tex](docs/task_cost_model.tex).
- **Pareto frontiers** – which providers are non-dominated across task cost, latency, throughput,
  cache hit and uptime at once, and which models no cheaper model out-scores on benchmark
  quality, with the **efficient point** where marginal quality per dollar starts to fall off.
- **Routing** – "I need at least this much intelligence": the model and provider to call, chosen
  from the frontier rather than from a price list.

We also solve the efficient frontier across models, joining OpenRouter's 24-hour realized
pricing (which changes as providers come and go) with Artificial Analysis benchmark scores, so
the cost side of "quality per dollar" reflects what the model costs to call today, not its
list price.

## Installation

```bash
git clone https://github.com/deep-blue-dark-red/openrouter-frontier.git && cd openrouter-frontier
uv venv && source .venv/bin/activate
uv pip install -e .            # installs the `openrouter-frontier` / `or-frontier` CLI
uv pip install -e '.[dev]'     # adds pytest
```

The repo-root scripts (`./score_providers.py`, `./openrouter-tui`, …) locate the local
`.venv` themselves, so they run without activating it. Symlink any of them onto your `$PATH`.

## Tools

| Tool                   | Question it answers                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `openrouter-tui`       | Interactive: browse the model frontier, drill into a model's providers                 |
| `score_providers.py`   | Which provider is cheapest for this model, all things considered?                      |
| `provider_frontier.py` | Which providers are non-dominated on cost, latency, TPS, cache hit, uptime?            |
| `model_frontier.py`    | Which models give the most benchmark quality per dollar? Where is the efficient point? |
| `model_router.py`      | I need at least this much intelligence: which model and provider should I call?        |
| `get_models.py`        | Search and inspect the 400+ model catalog                                              |
| `openrouter-frontier` | Rich-coloured CLI: `stats`, `score`, `cache`, `compare`, `search`                      |

All tools accept fuzzy model names: `zai/glm-5.3-flsh`, `z.ai/glm-5.3-flash`, and
`glm-5.3-flash` all resolve to `z-ai/glm-5.3-flash`.

### `openrouter-tui` — interactive explorer

```bash
./openrouter-tui
```

![openrouter-tui models view](docs/openrouter-tui.png)

- **Models view** – every benchmarked model, sorted by the cost vs. quality Pareto frontier:
  frontier models first from cheapest up, then dominated models by their gap to the frontier.
  `★ OPTIMAL` marks frontier models, `★ EFFICIENT` the frontier's efficient point (best quality
  gain per dollar), and `-N pts` a dominated model's score gap. Type to filter (`zai`,
  `claude37`, `gemini25` all work; punctuation is ignored). **Tab** switches between the
  intelligence and coding index; the active metric is shown in the status line.
- **Providers view** – **Enter** on a model ranks its endpoints by ProviderScore expected
  task cost. **Tab** or **a** toggles between the primary quantization and all variants.
- **Detail view** – **Enter** on a provider shows the task-cost decomposition (fixed, time,
  cached-read baseline, miss premium, failure premium), the perfect- and cold-cache bounds, the
  risk terms, the probability of migrating to the fallback, and the prices used.
- Navigation: Up/Down or Ctrl-P/N, PgUp/PgDn, mouse wheel and click, Esc/Backspace to go
  back (Esc clears the search first), Ctrl-C to quit. Runs in the alternate screen buffer so
  your scrollback is untouched.

### `score_providers.py` — expected task cost per provider

```bash
./score_providers.py z.ai/glm-5.3-flash --top 5                     # default: 300k context, 10k output, $20/hr
./score_providers.py z.ai/glm-5.3-flash -t 0                          # tokens and failures only
./score_providers.py z.ai/glm-5.3-flash --task-tokens 1000000 -a 4000  # longer task, bigger tool results
./score_providers.py z.ai/glm-5.3-flash --output-tokens 50000         # chattier model
./score_providers.py z.ai/glm-5.3-flash --routing order               # explicit provider order (return to primary)
./score_providers.py z.ai/glm-5.3-flash --cache cold                  # cold-cache bound
./score_providers.py z.ai/glm-5.3-flash --sigma-h 0.03 --lambda-par 1 # penalise being wrong about the hit rate
./score_providers.py z.ai/glm-5.3-flash --all-quants --json
```

```text
ProviderScore Task Cost: Z.ai: GLM 5.3 Flash (z-ai/glm-5.3-flash)
Task: 145 turns × (2000 new + 69 out) → 300k context, 10k output  •  Time: $20/hr  •  Routing: sticky  •  Miss: rewrite  •  Cache: aggregate
Quantization: primary
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Provider              Task $   Tokens $     Time $    Fail $    Miss $  CacheHit   E[TTFT]    TPS   Uptime   Read $/M   Miss $/M       $/M
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Parasail               $8.35      $1.10      $7.24   $0.0070   $0.4050     84.3%     2.99s     53    96.7%    $0.0300    $0.1500   $0.3814
Baseten               $10.08      $2.00      $8.07   $0.0077     $1.30     49.5%     1.13s     93    99.5%    $0.0300    $0.1500   $0.4604
Sail Research         $10.96      $1.09      $9.87   $0.0096   $0.3889     84.9%     4.29s     38    98.0%    $0.0300    $0.1500   $0.5006
NovitaAI              $11.27    $0.5608     $10.71   $0.0049   $0.2125     83.5%     3.41s     33    99.2%    $0.0150    $0.0750   $0.5147
Phala                 $11.55      $1.33     $10.22   $0.0106   $0.6287     75.6%     2.21s     42    98.8%    $0.0300    $0.1500   $0.5275
Z.ai                  $11.91    $0.5034     $11.40   $0.0053   $0.1552     88.0%     6.09s     32    98.8%    $0.0150    $0.0750   $0.5440
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Task $ = expected cost of the whole task on that endpoint (tokens + time + failures); lower is better. Miss $ = cache-miss premium. $/M = task cost per 1M submitted tokens (secondary). CacheHit = published 24h rate. E[TTFT] = lognormal mean of p50/p90.
```

`Task $` is the expected cost of the whole task: `Tokens $` (new tokens, output, and the
cached-read baseline plus `Miss $`, the cache-miss premium) + `Time $` + `Fail $`. The default
task appends 2000 tokens of user text and tool results per turn until the context reaches 300k,
generating 10k output tokens in total (145 turns of 69), with time at $20/hr and OpenRouter's
default sticky routing. `Time $` counts time to first token, decoding at the published
throughput, and re-prefilling the prefix after a cache miss at 100× the decode rate, which is
why a high hit rate can beat raw throughput. `$/M` is the same cost per 1M submitted
tokens and is secondary: it falls as tasks get longer while the bill rises. `E[TTFT]` is the
lognormal mean fitted to the published p50/p90.

By default only the model's primary quantization (usually `fp8`) is shown, matching the
variant OpenRouter prices on the web. Pass `--all-quants` to see community variants too.

### `provider_frontier.py` — provider Pareto frontier

Reports which providers are **non-dominated** across five objectives: expected task cost ↓,
expected time to first token ↓, throughput ↑, cache hit rate ↑, uptime ↑. It takes the same
task-profile flags as `score_providers.py`.

```bash
./provider_frontier.py z.ai/glm-5.3-flash
./provider_frontier.py z.ai/glm-5.3-flash --optimal-only
./provider_frontier.py --models --optimal-only      # catalog-wide: turn cost vs context vs cache pricing
```

```text
Provider Pareto Frontier: Z.ai: GLM 5.3 Flash (z-ai/glm-5.3-flash)
Task: 145 turns × (2000 new + 69 out) → 300k context, 10k output  •  Time: $20/hr  •  Routing: sticky  •  Miss: rewrite  •  Cache: aggregate
Objectives: Task cost ↓  E[TTFT] ↓  TPS ↑  CacheHit ↑  Uptime ↑  •  Quantization: primary
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Provider              Task $   Tokens $   E[TTFT]    TPS  CacheHit   Uptime  Pareto Frontier  Niche / Advantage               
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Parasail               $8.35      $1.10     2.99s     53     84.3%    96.7%  ★ OPTIMAL        Lowest Cost                     
Baseten               $10.08      $2.00     1.13s     93     49.5%    99.5%  ★ OPTIMAL        Highest TPS                     
Sail Research         $10.96      $1.09     4.29s     38     84.9%    98.0%  ★ OPTIMAL        Balanced Trade-off              
NovitaAI              $11.27    $0.5608     3.41s     33     83.5%    99.2%  ★ OPTIMAL        Balanced Trade-off              
Phala                 $11.55      $1.33     2.21s     42     75.6%    98.8%  ★ OPTIMAL        Balanced Trade-off              
Z.ai                  $11.91    $0.5034     6.09s     32     88.0%    98.8%  ★ OPTIMAL        Best Cache Hit                  
Modal                 $12.17      $1.70     0.50s     52     61.4%    99.9%  ★ OPTIMAL        Lowest Latency • Highest Uptime 
Morph                 $13.90      $1.22     2.10s     41     68.4%    94.2%  ★ OPTIMAL        Balanced Trade-off              
SiliconFlow           $17.80      $1.54     2.03s     31     67.3%    99.7%  ★ OPTIMAL        Balanced Trade-off              
NextBit               $19.33      $1.91     3.02s     42     53.2%    99.6%  Dominated        --                              
DeepInfra             $19.57      $1.03     1.27s     40     47.1%    99.4%  Dominated        --                              
Reka AI               $22.47      $2.40     1.76s     46     34.0%    99.8%  Dominated        --                              
GMICloud              $32.50    $0.8234     9.21s     21     63.2%    98.1%  Dominated        --                              
io.net                $77.45      $2.49     1.36s     12     30.8%    90.3%  Dominated        --                              
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
★ OPTIMAL marks non-dominated providers: no other provider is at least as good on every objective and better on one.
```

### `model_frontier.py` — cost vs. quality frontier

Joins OpenRouter's published Artificial Analysis indices (intelligence, coding, agentic) with
live pricing and finds the models no cheaper model out-scores. The **efficient point** is the
frontier model with the best normalised quality gain per dollar, the maximum-gradient point of
the frontier.

```bash
./model_frontier.py                                 # intelligence index vs list prompt price
./model_frontier.py --metric coding --all-models    # include dominated models with their gap
./model_frontier.py --price-source call -c 2000 -o 500
./model_frontier.py --format markdown
```

```text
OpenRouter Cost vs. Intelligence Pareto Frontier
8 non-dominated of 87 benchmarked models  •  price source: list
──────────────────────────────────────────────────────────────────────────────────────────────────
Model                        ID                             Intelligence Score  Cost ($/1M prompt)  Status
──────────────────────────────────────────────────────────────────────────────────────────────────
Ling 3.0 Flash               inclusionai/ling-3.0-flash                   37.8               $0.02  ON FRONTIER
Solar Pro 4                  upstage/solar-pro4                           41.6               $0.03  ON FRONTIER
GLM-5.3-Flash                z-ai/glm-5.3-flash                           57.5               $0.07  ← EFFICIENT
Gemini 3.8 Flash (high)      google/gemini-3.8-flash                      58.7               $0.75  ON FRONTIER
GLM-5.3 (max)                z-ai/glm-5.3                                 59.5               $1.40  ON FRONTIER
```

### `model_router.py` — model and provider for a required score

Picks a benchmarked model whose Artificial Analysis score meets a threshold, then that
model's best provider from the ProviderScore scorer. Two modes choose the model:

- `cheapest` – the cheapest model at or above the score.
- `efficient` – the efficient point of the cost/quality frontier built from the models at or
  above the score: the most quality gained per dollar once the level is met.

Models with no active provider are skipped in favour of the next best qualifying one.

```bash
./model_router.py 60                                # cheapest model with intelligence >= 60
./model_router.py 60 --mode efficient               # most quality per dollar above 60
./model_router.py 45 --metric coding --task-tokens 100000
./model_router.py 70 --json
```

```python
from model_router import route
r = route(60, metric="intelligence", mode="efficient")
r.model_id, r.provider.provider_slug, r.provider.task_cost_usd
```

### `get_models.py` — catalog search and inspection

```bash
./get_models.py z-ai/glm-5.3-flash               # detail card: context, pricing, description
./get_models.py -q flash --top 10
./get_models.py --creator anthropic --sort context
./get_models.py --caching --sort price --top 10  # only models with a cache-read price
./get_models.py --refresh                        # bypass the 1-hour catalog cache
```

### `openrouter-frontier` — Rich CLI

```bash
openrouter-frontier stats   z-ai/glm-5.3-flash --sort cache --top 5   # 24h metrics table
openrouter-frontier score   z-ai/glm-5.3-flash --routing order        # ProviderScore task-cost ranking
openrouter-frontier cache   z-ai/glm-5.3-flash novita                 # one provider's task-cost breakdown
openrouter-frontier compare z-ai/glm-5.3-flash z-ai novita deepinfra  # side by side
openrouter-frontier search  glm
```

`stats --sort` accepts `cache`, `score`, `token_cost`, `latency`, `tps`, `uptime`, `input`,
`output`, `tokens`, `share`, `name`. Every command takes `--json`.

## Scoring model

The unit of cost is a **task**, not a call. The full derivation, assumptions, and the
variance results are in [docs/task_cost_model.tex](docs/task_cost_model.tex); this is the
summary the code implements (`openrouter_frontier/scoring.py`).

**Task profile.** $N$ turns; each appends $a$ new prompt tokens and $o$ completion tokens, so
the transcript grows by $d = a + o$ per turn. At turn $k$ the prompt is a reusable prefix
$S_k = (k-1)d$ plus the $a$ new tokens. Defaults: $a = 2000$, a context of 300,000 tokens and
10,000 output tokens in total, giving $N = 145$ and $o = 69$. Output is a small share of an
agentic transcript; most of it is tool results.

**Prices** (USD per token; endpoints API values are already net of discount): $b$ ordinary
input, $r$ cache read, $w$ cache write ($w = b$ when unpublished), $c$ completion, $f$ per
request. On a miss the prefix is billed at $m = w$ (rewritten, default) or $m = b$ (processed).
An endpoint with no read price has no cache: $r = w = b$, $h = 0$.

**Inputs.** $h$ is the published 24-hour token-weighted cache hit rate, used as observed (it is
a measurement over millions of requests, so no prior is applied; `--cache cold|assumed`
override it). $u$ is the published uptime. Time to first token and seconds per output token
are lognormal means fitted to the published p50 and p90. Prompt processing runs at
$g_p = 100\, g$, so a missed prefix costs $S_k / g_p$ seconds to re-prefill. Missing latency, throughput, or
uptime is imputed as the worst observed among the model's endpoints.

**Turn cost on the primary endpoint** (cached fraction $H_k$ with mean $h$, $v' = v/3600$):

$$
X_k^{\text{ok}} = f + w\,a + c\,o + v'\big(\mathbb{E}[\ell] + o\,\mathbb{E}[s]\big) + S_k\big[m - H_k (m - r)\big] + v'\,(1 - H_k)\,S_k / g_p,
\qquad
\mathbb{E}[X_k^{\text{ok}}] = \alpha + \pi S_k,\ \ \pi = h r + (1-h)(m + v'/g_p) .
$$

**Failure.** With probability $q = 1 - u$ the request fails: the caller has waited
$\mathbb{E}[\ell]$ for nothing and the turn is served cold by a fallback $B$ (default: the
endpoint itself, cold), every token written at $w_B$. Under `--routing order` the next turn
returns to the primary and pays a small return penalty $h(m-r)d$; under `--routing sticky`
(OpenRouter's default) the task stays on the fallback from then on.

**Task cost.** $X_{\text{task}} = \sum_{k=1}^{N} X_k$. Under `order`,

$$
\mathbb{E}[X_{\text{task}}] = N A_0 + \Pi\, d\,\frac{N(N-1)}{2} + q\,h\,(m-r)\,d\,(N-1),
\qquad \Pi = (1-q)\,\pi + q\,w_B ,
$$

and the quadratic term splits into a cached-read baseline $r$, a miss premium
$(1-q)(1-h)(m-r)$ in dollars plus $(1-q)(1-h)\,v'/g_p$ in re-prefill time, and a failure
premium $q(w_B - r)$ plus the cold prefill, each times $d\,N(N-1)/2$. Under
`sticky` the code evaluates the geometric mixture over the migration time turn by turn; the
probability the task finishes on the fallback is $1-(1-q)^N$, which at $N = 145$ is already
77% for an endpoint with 99% uptime.

**Risk.** The process bound $\sigma_{\text{proc}}$ (Bernoulli misses plus the failure mixture
term) grows as $N^{3/2}$. The parameter term $\sigma_{\text{par}} = (1-q)(m-r+v'/g_p)\sigma_h\, d\,N(N-1)/2$
grows as $N^2$ for a caller-supplied $\sigma_h$ (drift or workload mismatch). Endpoints are
ranked by $J = \mathbb{E}[X_{\text{task}}] + \lambda_{\text{proc}}\sigma_{\text{proc}} + \lambda_{\text{par}}\sigma_{\text{par}}$,
with both $\lambda$ zero by default.

## Pareto analysis

**Dominance.** Provider $b$ dominates $a$ if $b$ is at least as good on every objective and
strictly better on at least one. Providers that nobody dominates form the frontier. A missing
metric is treated as the worst possible value, so an endpoint with no latency data can never
win on latency. The provider frontier uses expected task cost, expected TTFT, throughput, $h$,
and uptime; the catalog frontier (`provider_frontier.py --models`) uses turn cost, context
length, cache-read price, and completion price.

**Cost vs. quality frontier** (`model_frontier.py`, TUI). Models are sorted by ascending cost; a
model is on the frontier when its score exceeds every cheaper model's score. For a dominated
model, the reported gap is the best frontier score available at or below its cost minus its own
score.

**Efficient point.** Both axes are min–max normalised over the frontier to $[0,1]$. The
efficient point is the frontier model maximising $\hat{s} - \hat{c}$, i.e. the point furthest
above the chord joining the cheapest and the best-scoring frontier models. That is where the
marginal quality gained per dollar starts to fall off.

## Python API

```python
from openrouter_frontier import (
    ScoringConfig, score_model_providers, get_model_stats,
    Objective, pareto_mask,
)

# Rank providers by expected task cost (300k-token task, $20/hr, sticky routing by default).
for s in score_model_providers("z-ai/glm-5.3-flash")[:3]:
    print(f"{s.provider_name:<12} {s.formatted_task_cost} per task  time {s.formatted_time_cost}  cache hit {s.formatted_cache_hit_rate}")

# A 1M-token agentic task with big tool results, explicit provider order, tokens only.
cfg = ScoringConfig(new_tokens_per_turn=6000, completion_tokens=800, task_tokens=1_000_000,
                    time_value_usd_per_hour=0, routing="order")
best = score_model_providers("z-ai/glm-5.3-flash", config=cfg)[0]
best.task_cost_usd, best.miss_premium_usd, best.migration_probability

# Raw 24h stats, sorted however you like.
stats = get_model_stats("glm-5.3-flash")
fastest = stats.sort_by("latency")[0]

# Generic N-objective Pareto mask over your own objects.
scores = stats.score_providers()
mask = pareto_mask(scores, [
    Objective(lambda s: s.task_cost_usd, minimize=True),
    Objective(lambda s: s.ttft_seconds, minimize=True),
    Objective(lambda s: s.uptime_pct, minimize=False),
])
frontier = [s.provider_name for s, ok in zip(scores, mask) if ok]
```

## Data sources

| Endpoint                                               | Provides                                                           |
| ------------------------------------------------------ | ------------------------------------------------------------------ |
| `api/frontend/v1/stats/effective-pricing`              | per-endpoint 24h cache hit rate, effective prices, token volume    |
| model page RSC payload (`openrouter.ai/models/<slug>`) | p50/p90 latency and throughput, 24h uptime                         |
| `api/v1/models/<slug>/endpoints`                       | list pricing incl. cache read/write, quantization, uptime fallback |
| `api/v1/models`                                        | catalog: context length, modality, list pricing                    |
| `api/frontend/v1/rankings/benchmarks`                  | Artificial Analysis intelligence / coding / agentic indices        |

Connections are forced to IPv4 to avoid a ~10 s IPv6 stall on macOS.

## Development

```bash
uv pip install -e '.[dev]'
pytest                      # unit tests for the scoring and Pareto math
```

## License

MIT
