# Agent Debugger — Live

Real-time failure detection and **blocking** for LangChain/LangGraph agents. Attach one callback handler and catch the failures that don't throw errors — the ones that return `status: success` while lying to your users.

**See it catch a lie in 60 seconds, no API keys needed:**

```bash
git clone https://github.com/arun-molugu/agent-debugger-live.git
cd agent-debugger-live
pip install langchain-core openai
python3 demo_multi_agent_propagation.py
```

You'll watch a two-node agent pipeline fail the way real ones do: the shipping API dies silently, node 1 hallucinates a tracking number, node 2 trusts it and drafts a confident email to the customer — and the debugger catches the hallucination, catches it **propagating** into the second node, and blocks the email before it ships.

## Why this exists

Most agent failures don't throw errors. CI/CD can't catch them because the code didn't fail — the *agent* did. It called a tool, the tool errored, and the agent confidently reported success anyway. This catches that failure class live, in-process, as it happens — and can stop the response before your user sees it.

## What it catches (17 detectors)

**Deterministic Layer 1** — zero network calls, runs in-process at ~21k executions/sec:

- **HALLUCINATION** — agent claimed success right after a tool call returned empty
- **CROSS_NODE_PROPAGATION** — a downstream node repeated an upstream node's flagged claim as verified fact (multi-agent graphs)
- **NUMERICAL_MISMATCH** — agent reports a different number than the tool actually returned
- **BOOKING_CLAIM_WITHOUT_TOOL** — agent claims a booking/purchase happened with no tool call
- **HALLUCINATED_RETRY** — agent claims a retry succeeded but no retry call happened
- **CONTEXT_DROP** — agent contradicts a fact it stated earlier in the same session
- **UNVERIFIABLE_ASSERTION** — agent claims an internal mechanism (retry logic, validation, rollback) ran, with no observable evidence in the trace
- **THEMATIC_OSCILLATION** — a validator/repair loop keeps raising the same issue in different words, never converging
- **TOOL_AVOIDANCE** — agent answered a real-time-data query without calling any tool
- **ACTION_SKIPPED** — a tool was called and returned nothing, silently
- **GOAL_ABANDONMENT** — agent signals intent to continue but the task ends there
- **DATE_MISINTERPRETATION** — tool scheduled one date, agent confirmed a different one
- **PERMISSION_FAILURE** — tool returned an authorization error
- **RETRY_LOOP** — repeated retries with no stopping condition
- **RISK_FLAG** / **CRITICAL_SYSTEM_FAILURE** — warning and critical status signals

**Gated Layer 2 (optional, off by default)** — when Layer 1 stays silent, a GPT-4o-mini call checks for subtler contradictions keyword matching structurally can't see (**SEMANTIC_ANOMALY**). Requires `OPENAI_API_KEY`; only fires when deterministic checks find nothing.

**Blocking**: with `block_on_critical=True`, any critical-severity catch prevents the response from being returned — `should_block()` / `get_safe_response()` — so the failure is stopped, not just logged.

## Local-first by design

Layer 1 makes **zero network calls**. Your traces never leave your process — no cloud platform, no third-party trace storage, nothing to add to your compliance scope. The semantic layer is the only component that calls out, and it's off by default.

## Integrate with a real agent (~10 lines)

```python
from langchain.agents import create_agent
from agent_debugger_live import LiveWatchHandler

handler = LiveWatchHandler(block_on_critical=True)

agent = create_agent(model="gpt-4o-mini", tools=[your_tools],
                     system_prompt="You are a helpful assistant.")

result = agent.invoke(
    {"messages": [{"role": "user", "content": user_message}]},
    config={"callbacks": [handler]},
)
final = result["messages"][-1].content
handler.check_agent_response(final)          # add node="node_name" in multi-node graphs

safe_output = handler.get_safe_response(final)   # blocked responses never reach the user
```

## More demos

```bash
# Live LangGraph multi-node validator/repair loop (needs OPENAI_API_KEY):
python3 example_validator_repair.py

# Real Stripe test-mode refund hallucination, caught and blocked
# (needs OPENAI_API_KEY + STRIPE_API_KEY test key):
python3 real_agent_demo.py
```

Full setup for the live demos: `pip install -r requirements.txt`

## Tested against reality, not mocks

20/20 hallucination catch rate on real GPT-4o-mini output across two temperature settings; verified end-to-end against real Stripe test-mode API calls via LangChain's `create_agent`; thematic oscillation confirmed on real multi-round LangGraph loops. Real bugs found and fixed through this testing (negation false-positives, `ToolMessage` compatibility, ID-hyphen number parsing) — the fix history is in the commits.

## Honest limitations

Early-stage, not production infrastructure. Detection is deterministic keyword/topic/number analysis, not full semantic understanding — Layer 2 exists precisely because of that. Tested on LangChain/LangGraph with GPT-4o-mini; other frameworks and model families are unverified. Cross-node tracking currently requires you to pass `node=` explicitly. No persistent history across sessions, no Slack/webhook alerting yet.

## Feedback

If you run this on your own agent and it catches — or misses — something interesting, I genuinely want to hear about it. Open an issue, or DM me.
