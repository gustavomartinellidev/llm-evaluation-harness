# LLM Evaluation Harness — LangGraph PoC

A multi-node state graph that benchmarks generative models in parallel, captures quality, latency, tokens and cost, and produces a structured decision memo with an adopt, watch, or avoid recommendation.

I built this while taking [LangChain Academy's *Introduction to LangGraph*](https://academy.langchain.com/), to practice the framework on a real problem: when an applied AI team picks a model for a task, the answer is always a trade-off across quality, latency, and cost. This harness automates the comparison.

---

## Architecture

> *Replace this block with the PNG exported from the notebook's graph visualization cell (`graph.get_graph().draw_mermaid_png()`).*

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
| `Send` API | `dispatch_models` | Dynamic fan-out, one worker per item in the input list, decided at runtime |
| Conditional edges | After `judge_outputs` | Routes to the memo or to the error handler based on judge outcome |
| LLM-as-judge | `judge_outputs` | Quality assessment when there is no ground truth |
| Structured output parsing | `judge_outputs` → memo | JSON contract between the judge LLM and the memo writer |
| Retry with exponential backoff | `run_model` | Tolerates `429 RESOURCE_EXHAUSTED` from rate-limited tiers |

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
| `gpt-5.4-nano` | ✅ | 3.75 | 125 | 0.1636 | 8/10 |
| `gpt-5.4-mini` | ✅ | 3.60 | 129 | 0.6082 | 9/10 |
| `gpt-5.4`      | ✅ | 5.25 | 119 | 1.8775 | 10/10 |

#### Quality Rationale

- **`gpt-5.4-nano`** (8/10): Accurate and clear, but slightly generic and less explicit about emerging-market constraints and the specific role of intermediary banks in correspondent settlement.
- **`gpt-5.4-mini`** (9/10): Well-balanced and precise, covering currency conversion, FX spreads, and intermediary banks with good clarity and relevant emerging-market context.
- **`gpt-5.4`** (10/10): Fully addresses the prompt in exactly three sentences with accurate explanation of conversion mechanics, FX spreads, and the role of intermediary banks in cross-border emerging-market payments.

#### Recommendation

**Adopt:** `gpt-5.4`

- Quality: 10/10
- Latency: 5.25s
- Cost: USD 1.8775 per 1k runs

*Recommendation rule: highest quality score; ties broken by lowest cost.*

---

*(Numbers above come from one real run against OpenAI's API. They vary per run and per pricing snapshot. A practical operator would often prefer `gpt-5.4-mini` here, since the quality gap is small and the cost is roughly 3× lower.)*

---

## Run it

You need:
- An OpenAI API key: https://platform.openai.com/api-keys
- A Google Colab account, or a local Jupyter environment

New OpenAI accounts get USD 5 in free credits valid for 3 months, which is enough to run this harness many times over.

### In Colab

1. **File → Upload notebook**, then select `LLM_Evaluation_Harness_LangGraph.ipynb`.
2. Open the **Secrets** panel (🔑 icon, left sidebar), add a secret named `OPENAI_API_KEY` with your key as the value, and toggle notebook access on.
3. **Runtime → Run all**.

### Locally

```bash
pip install langgraph langchain-openai openai
export OPENAI_API_KEY="your-key-here"
jupyter notebook LLM_Evaluation_Harness_LangGraph.ipynb
```

Replace the `from google.colab import userdata` line with `os.environ["OPENAI_API_KEY"]` directly.

### Customize

The two inputs that change in the final code cell:

```python
result = graph.invoke({
    "task": "<your prompt here>",
    "models": ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"],
    ...
})
```

---

## Extension ideas

Natural next steps for the harness:

- **Multi-task batches.** Turn `task` into `List[str]`, then reduce judge scores across tasks for a richer comparison.
- **Multiple judges.** Run two judge models in parallel and report inter-judge agreement, to surface evaluation noise.
- **Regression tracking.** Persist results with a timestamp, diff against the last run to catch quality drift after model updates.
- **Cost-aware routing.** Add a router node that picks the cheapest model meeting a quality threshold.
- **Provider-agnostic.** Swap `ChatOpenAI` for `ChatAnthropic`, `ChatGoogleGenerativeAI`, or local models via `ChatOllama`.
- **Streaming and budgets.** Add a budget node that cancels in-flight workers once total cost crosses a threshold.

---

## Caveats

- **Pricing.** The `MODEL_PRICING` dict in the notebook is a snapshot. Confirm current rates at https://openai.com/api/pricing before reporting costs externally.
- **Free-tier rate limits.** OpenAI's free credits come with per-model rate limits, and consecutive runs on flagship models can return `429 RESOURCE_EXHAUSTED`. The worker handles this with exponential backoff (4s, 8s, 16s) and reports failures cleanly in the comparison table.
- **LLM-as-judge is not ground truth.** Judge scores are useful for *relative* comparison across models on the same task. They are not a substitute for human evaluation on production-critical decisions.
- **Single-shot measurements are noisy.** For serious benchmarking, average latency and quality over N runs. This harness is a starting point, not a regression suite.

---

## About

Built while taking [LangChain Academy's *Introduction to LangGraph*](https://academy.langchain.com/). The course concepts are real (Send, reducers, conditional edges, LLM-as-judge), but I wanted to apply them to a problem an applied AI team actually faces, not a tutorial example.

— Gustavo Gobi Martinelli

## License

MIT
