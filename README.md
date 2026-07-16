# Agent Debugger — Live

Real-time failure detection for LangChain/LangGraph agents. No pasting, no dashboards — attach it directly to your agent and get flagged the instant something goes wrong.

## What it catches right now

- **ACTION_SKIPPED** — a tool was called and returned nothing, silently
- **HALLUCINATION** — agent claimed success right after a tool call returned empty
- **THEMATIC_OSCILLATION** — a validator/repair loop keeps raising the same underlying issue in different words, never converging
- **TOOL_AVOIDANCE** — agent answered a query requiring real-time data without calling any tool
- **GOAL_ABANDONMENT** — agent's response signals intent to continue but the task ends there
- **DATE_MISINTERPRETATION** — tool scheduled one date, agent confirmed a different one to the user
- **NUMERICAL_MISMATCH** — agent reports a different number than what the tool actually returned
- **PERMISSION_FAILURE** — tool returned an authorization/permission error
- **RETRY_LOOP** — agent is retrying the same action repeatedly with no stopping condition
- **HALLUCINATED_RETRY** — agent claims a retry succeeded but no retry tool call actually happened
- **RISK_FLAG** — step reported a warning-level risk signal
- **CRITICAL_SYSTEM_FAILURE** — system reported a critical error
- **BOOKING_CLAIM_WITHOUT_TOOL** — agent claims a booking/purchase happened with no tool call
- **CONTEXT_DROP** — agent contradicts a fact it stated earlier in the same session
- **UNVERIFIABLE_ASSERTION** — agent claims an internal mechanism (retry logic, validation, rollback) ran, with no observable evidence in the trace

Can also **block** a response before it's returned when a critical-severity failure is detected, instead of only flagging it after the fact.

## Why this exists

Most agent failures don't throw errors. They return `status: success` and look completely normal — while doing the wrong thing. This catches that failure class, live, as it happens.

Tested against real GPT-4o-mini output (not scripted mocks): 20/20 hallucination catch rate across two temperature settings, and multiple independent confirmations of thematic oscillation detection on a real multi-round LangGraph validator/repair loop.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Quick start

```python
from agent_debugger_live import LiveWatchHandler

handler = LiveWatchHandler(block_on_critical=True)

result = my_tool.invoke({"arg": "value"}, config={"callbacks": [handler]})
handler.check_agent_response(agent_reply_text)

if handler.should_block():
    print("Response blocked — do not send to user")

print(handler.summary())
```

## Try the demo

```bash
python3 example_validator_repair.py
```

## Honest limitations

Early prototype, not production infrastructure. Detection is keyword and topic-overlap based, not full semantic understanding. Tested on one framework (LangChain/LangGraph), one model family (GPT-4o-mini), and a narrow set of scenarios. Does not yet support multi-agent coordination, persistent history across sessions, or alerting integrations (Slack/webhooks).

## Feedback

If you try this on your own agent and it catches — or misses — something interesting, I'd genuinely like to hear about it.
