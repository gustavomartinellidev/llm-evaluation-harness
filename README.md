# LLM Evaluation Harness — LangGraph

A small, instrumented harness that benchmarks multiple LLMs in parallel on the same task and produces a structured **decision memo** with an `adopt` recommendation.

Built to practice [LangGraph](https://langchain-ai.github.io/langgraph/) patterns — dynamic fan-out, reducers, conditional edges, LLM-as-judge — on a problem an applied AI team actually faces: *given a task, which model should we use?*

---

## Why this exists

Choosing an LLM for a specific use case is a trade-off across **quality, latency, and cost**, and the right answer changes per task. This harness automates the comparison: same prompt across *N* models, captured side-by-side with a quality score, ranked, and synthesized into a memo you can paste into a tech-decision document.

It is intentionally small (one notebook, ~200 lines) — the goal is the *shape* of the workflow, not the breadth of providers.

---

## Architecture

> _Replace this block with the PNG exported from the notebook's graph visualization cell (`graph.get_graph().draw_mermaid_png()`)._

```
START
  │
  ▼
dispatch_models  ── (Send → fan-out one worker per model)
  │
  ├──► run_model ──┐
  ├──► run_model ──┤  (parallel)
  └──► run_model ──┤
                   ▼
              judge_outputs   (LLM-as-judge: scores 1–10 with rationale)
                   │
                   ├──[ judge succeeded ]──► write_decision_memo ──► END
                   │
                   └──[ no successful runs ]──► error_handler ──────► END
```

### LangGraph patterns used

| Pattern | Where | Why |
|---|---|---|
| `StateGraph` + `TypedDict` | `EvaluationState` | Typed shared state across nodes |
| `Annotated[List[dict], add]` reducer | `results` field | Merges parallel worker outputs without races |
| `Send` API | `dispatch_models` | Dynamic fan-out — one worker per item in the input list, decided at runtime |
| Conditional edges | After `judge_outputs` | Routes to memo or error handler based on judge outcome |
| LLM-as-judge | `judge_outputs` | Quality assessment when ground truth isn't available |
| Structured output parsing | `judge_outputs` → memo | JSON contract between judge LLM and memo writer |

---

## Example output

For the task:

> *"In 3 sentences, explain how distributed payment systems handle currency conversion for cross-border transactions in emerging markets, including the role of FX spreads and intermediary banks."*

The harness produces a markdown memo like this:

---

### Decision Memo — LLM Evaluation

**Task evaluated:** *(as above)*

#### Comparison Table

| Model | Status | Latency (s) | Out tokens | Cost / 1k runs (USD) | Quality |
|---|---|---|---|---|---|
| `gemini-2.0-flash` | ✅ | 1.42 | 87 | $0.0383 | 7/10 |
| `gemini-2.5-flash` | ✅ | 2.18 | 94 | $0.2632 | 9/10 |
| `gemini-2.5-pro` | ✅ | 4.91 | 102 | $1.1450 | 9/10 |

#### Quality Rationale

- **`gemini-2.0-flash`** (7/10): Covers FX spreads but omits intermediary bank role; explanation slightly compressed.
- **`gemini-2.5-flash`** (9/10): Accurate, complete, and concise; correctly attributes spread capture to liquidity providers.
- **`gemini-2.5-pro`** (9/10): Equivalent quality to Flash but ~2× latency and ~4× cost on this task.

#### Recommendation

**Adopt:** `gemini-2.5-flash` — best quality/cost balance for this task.

- Quality: 9/10
- Latency: 2.18s
- Cost: $0.2632 per 1k runs

*Recommendation rule: highest quality score; ties broken by lowest cost.*

---

*(Numbers above are illustrative. Actual results vary per run and per pricing snapshot.)*

---

## Run it

You need:
- A Google AI Studio API key — free tier is sufficient: https://aistudio.google.com/apikey
- A Google Colab account (or a local Jupyter environment)

### In Colab

1. **File → Upload notebook** → select `LLM_Evaluation_Harness_LangGraph.ipynb`
2. Open the **Secrets** panel (🔑 icon, left sidebar), add `GOOGLE_API_KEY`, toggle notebook access on
3. **Runtime → Run all**

### Locally

```bash
pip install langgraph langchain-google-genai
export GOOGLE_API_KEY="your-key-here"
jupyter notebook LLM_Evaluation_Harness_LangGraph.ipynb
```

Replace the `from google.colab import userdata` line with `os.environ["GOOGLE_API_KEY"]` directly.

### Customize

The two inputs you'll typically change live in the final code cell:

```python
result = graph.invoke({
    "task": "<your prompt here>",
    "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    ...
})
```

---

## Extension ideas

The harness is intentionally small. Natural next steps:

- **Multi-task batches.** Turn `task` into `List[str]`; reduce judge scores across tasks for a richer comparison.
- **Multiple judges.** Run two judge models in parallel; report inter-judge agreement to surface evaluation noise.
- **Regression tracking.** Persist results with a timestamp; diff against last run to catch quality drift after model updates.
- **Cost-aware routing.** Add a router node that selects the cheapest model meeting a quality threshold.
- **Provider-agnostic.** Swap `ChatGoogleGenerativeAI` for `ChatAnthropic`, `ChatOpenAI`, or local models via `ChatOllama`.
- **Streaming + budgets.** Add a budget node that cancels in-flight workers once total cost crosses a threshold.

---

## Caveats

- **Pricing.** The `MODEL_PRICING` dict in the notebook is a snapshot — verify current rates at https://ai.google.dev/pricing before reporting costs externally.
- **LLM-as-judge is not ground truth.** Judge scores are useful for *relative* comparison across models on the same task; they are not a substitute for human evaluation on production-critical decisions.
- **Single-shot measurements are noisy.** For serious benchmarking, average latency and quality over N runs — this harness is a starting point, not a regression suite.

---

## About

Built while taking [LangChain Academy's *Introduction to LangGraph*](https://academy.langchain.com/) — turning the course concepts into a workflow that mirrors real applied-AI evaluation decisions.

— Gustavo Gobi Martinelli

## License

MIT
