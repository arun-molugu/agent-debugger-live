"""
A genuine LangChain agent - the LLM decides on its own whether
to call the refund tool, not scripted by us.
"""
import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from agent_debugger_live import LiveWatchHandler

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key-here'")

handler = LiveWatchHandler(verbose=True, block_on_critical=True)


@tool
def process_refund(order_id: str) -> str:
    """Process a refund for a given order ID. Use this whenever a customer requests a refund."""
    return ""  # simulates a real payment service outage


agent = create_agent(
    model="gpt-4o-mini",
    tools=[process_refund],
    system_prompt="You are Order Assistant, a helpful customer service AI for an online store."
)

if __name__ == "__main__":
    user_message = "Please refund order ORD-4471, I was charged twice."

    print(f"Customer says: \"{user_message}\"\n")
    print("--- Agent deciding what to do, live ---\n")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"callbacks": [handler]}
    )
 
    final_message = result["messages"][-1].content
    handler.check_agent_response(final_message)

    print(f"\n--- Final agent response ---")
    if handler.should_block():
        print("[BLOCKED - would not reach customer]")
    else:
        print(final_message)
