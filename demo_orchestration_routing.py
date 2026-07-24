"""
REAL DYNAMIC ORCHESTRATION - a live supervisor agent decides, at runtime,
which specialist to route to. This is NOT a fixed pipeline (compare to
demo_pipeline_*.py) - the path through the graph is a live decision made
by a real model call, and can differ every run.

The scenario: a customer support supervisor reads an incoming query and
must route it to either the billing specialist or the shipping specialist.
Whichever one it is NOT routed to never even runs - a structurally
different failure surface than a linear chain, where every node always
ran. Here, an entire wrong branch can execute instead of the right one.

New detector: ROUTING_MISMATCH - after the supervisor picks a route, we
compare the query's real topic against the domain each specialist covers.
If the router sent a billing question to the shipping specialist (or vice
versa), that's a routing failure a linear-chain debugger structurally
cannot see, because there's no "wrong output" from the specialist - it
answers ably within ITS domain, just for the wrong query.

Requires OPENAI_API_KEY. ~1-2 cents per run. Real model calls throughout.
"""

import os
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from agent_debugger_live import LiveWatchHandler

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
handler = LiveWatchHandler(verbose=True, block_on_critical=True)

# What each specialist actually covers - used to check the supervisor's
# routing decision against the query's real topic, not to run the agents.
ROUTE_DOMAINS = {
    "billing":  {"refund", "charge", "payment", "invoice", "billing",
                "charged", "subscription", "card", "receipt", "price"},
    "shipping": {"tracking", "shipment", "delivery", "package", "carrier",
                "shipping", "arrive", "order", "address", "delayed"},
}

class OrchestrationState(TypedDict):
    query: str
    route: str
    response: str

# --- Supervisor: a REAL live routing decision, not scripted ------------
def supervisor_node(state: OrchestrationState) -> dict:
    prompt = (
        f"You are a routing supervisor for customer support. Read this "
        f"customer query and decide which specialist should handle it: "
        f"'billing' (for payments, charges, refunds, invoices) or "
        f"'shipping' (for tracking, delivery, packages, carriers). "
        f"Customer query: \"{state['query']}\"\n"
        f"Reply with exactly one word: billing or shipping."
    )
    decision = llm.invoke(prompt).content.strip().lower()
    route = "billing" if "billing" in decision else "shipping"
    print(f"\n[supervisor] query: \"{state['query']}\"")
    print(f"[supervisor] routed to: {route}")

    # NEW CHECK: does the chosen route's domain actually match the query?
    handler.check_routing_decision(state["query"], route, ROUTE_DOMAINS)
    return {"route": route}

# --- Specialists: each a REAL agent, only ONE of these two ever runs ---
def billing_node(state: OrchestrationState) -> dict:
    prompt = f"You are a billing specialist. Answer this customer query in 1-2 sentences: \"{state['query']}\""
    resp = llm.invoke(prompt).content
    print(f"\n[billing_agent] {resp}")
    handler.check_agent_response(resp, node="billing_agent")
    return {"response": resp}

def shipping_node(state: OrchestrationState) -> dict:
    prompt = f"You are a shipping specialist. Answer this customer query in 1-2 sentences: \"{state['query']}\""
    resp = llm.invoke(prompt).content
    print(f"\n[shipping_agent] {resp}")
    handler.check_agent_response(resp, node="shipping_agent")
    return {"response": resp}

def route_selector(state: OrchestrationState) -> Literal["billing", "shipping"]:
    return state["route"]  # the actual dynamic branch decision

# --- Wire the real dynamic graph ---------------------------------------
graph = StateGraph(OrchestrationState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("billing", billing_node)
graph.add_node("shipping", shipping_node)
graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_selector,
                            {"billing": "billing", "shipping": "shipping"})
graph.add_edge("billing", END)
graph.add_edge("shipping", END)
app = graph.compile()

if __name__ == "__main__":
    # A genuinely tempting query: real billing question, but loaded with
    # shipping-flavored distractor words ("package", "tracking", "shipped")
    # to actually test whether the router gets pulled the wrong way.
    # Verified: billing scores 5 vs shipping's 2 on keyword overlap, so if
    # the router picks shipping anyway, that's a real, defensible catch -
    # not a keyword-scoring artifact.
    test_query = ("My subscription payment was charged twice this month and "
                  "I'd like a refund on the extra charge, even though I can "
                  "see my package tracking shows it already shipped.")

    print("=" * 66)
    print("  REAL DYNAMIC ORCHESTRATION - supervisor decides live")
    print("=" * 66)

    final = app.invoke({"query": test_query, "route": "", "response": ""})

    print("\n" + "=" * 66)
    print("  WOULD THIS RESPONSE HAVE SHIPPED?")
    print("=" * 66)
    print(handler.get_safe_response(final["response"]))

    print("\n--- Diagnostic summary ---")
    for f in handler.summary()["flags"]:
        print(f"  [{f['severity'].upper()}] {f['type']}: {f['message']}")
