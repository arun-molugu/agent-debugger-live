"""
Side-by-side demo: same broken agent, with and without protection.
"""
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from agent_debugger_live import LiveWatchHandler

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key-here'")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


@tool
def process_refund(order_id: str) -> str:
    """Process a refund for the given order ID."""
    return ""  # simulates a real payment service outage


def run_unprotected_agent(order_id: str) -> str:
    tool_result = process_refund.invoke({"order_id": order_id})
    prompt = f"""You are Order Assistant, a helpful customer service AI. You have access to a refund processing system.
A customer just asked: "Please refund order {order_id}, I was charged twice."
System result: {tool_result}
Reply to the customer now."""
    response = llm.invoke(prompt)
    return response.content


def run_protected_agent(order_id: str) -> str:
    handler = LiveWatchHandler(verbose=False, block_on_critical=True)
    tool_result = process_refund.invoke({"order_id": order_id}, config={"callbacks": [handler]})
    prompt = f"""You are Order Assistant, a helpful customer service AI. You have access to a refund processing system.
A customer just asked: "Please refund order {order_id}, I was charged twice."
System result: {tool_result}
Reply to the customer now."""
    response = llm.invoke(prompt)
    agent_text = response.content
    handler.check_agent_response(agent_text)

    if handler.should_block():
        flags = handler.summary()["flags"]
        last_flag = flags[-1]
        return (
            f"[RESPONSE BLOCKED BEFORE REACHING CUSTOMER]\n"
            f"  What the agent almost said: \"{agent_text}\"\n"
            f"  Why it was blocked: {last_flag['message']}\n"
            f"  Actual system status: refund tool returned no data (silent failure)"
        )
    return agent_text


if __name__ == "__main__":
    order_id = "ORD-4471"

    print("=" * 65)
    print("SAME AGENT. SAME FAILURE. Payment service is down.")
    print("=" * 65)

    print("\n--- WITHOUT Agent Debugger ---")
    print(f"\nCustomer sees:\n  \"{run_unprotected_agent(order_id)}\"")

    print("\n--- WITH Agent Debugger ---")
    print(f"\n{run_protected_agent(order_id)}")

    print("\n" + "=" * 65)
