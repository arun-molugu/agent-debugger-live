"""
CROSS-NODE FAILURE PROPAGATION DEMO — zero API keys required.

The scenario (a real multi-agent pattern):
  Node 1 "order_lookup"   — calls the shipping API. The API fails silently.
                            The node HALLUCINATES a tracking number anyway.
  Node 2 "customer_email" — never sees the API failure. It trusts node 1's
                            summary and writes a confident email to the
                            customer containing the fabricated tracking number.

Per-node monitoring catches node 1's lie. But in a real graph, node 2's
email is what actually ships to the customer — and node 2 did nothing
"wrong" locally. The failure COMPOUNDED.

Agent Debugger catches both:
  1. HALLUCINATION at node 1  (success claim after empty tool output)
  2. CROSS_NODE_PROPAGATION at node 2  (repeated a tainted claim as verified)
  3. The final email is BLOCKED before it reaches the customer.

Run it:  python demo_multi_agent_propagation.py
"""

from agent_debugger_live import LiveWatchHandler

handler = LiveWatchHandler(verbose=True, block_on_critical=True)

print("=" * 62)
print("  MULTI-AGENT PIPELINE: order_lookup  ->  customer_email")
print("=" * 62)

# ---------------------------------------------------------------
# NODE 1: order_lookup calls the shipping API. It fails silently.
# ---------------------------------------------------------------
print("\n[node: order_lookup] calling shipping API for ORD-4471...")
shipping_api_output = ""  # real-world pattern: API error swallowed by a try/except
handler.on_tool_end(shipping_api_output)

# The node hallucinates a status anyway:
lookup_summary = (
    "Order ORD-4471 shipped successfully via FedEx. "
    "Tracking number: 784512963. Expected delivery in 2 days."
)
print(f"[node: order_lookup] output: {lookup_summary}")
handler.check_agent_response(lookup_summary, node="order_lookup")

# ---------------------------------------------------------------
# NODE 2: customer_email consumes node 1's summary as ground truth.
# It has no idea the shipping API ever failed.
# ---------------------------------------------------------------
customer_email = (
    "Hi Sarah, great news! Your order ORD-4471 is confirmed and on its way "
    "with FedEx. You can track it with tracking number 784512963. "
    "It should arrive within 2 days. Thanks for shopping with us!"
)
print(f"\n[node: customer_email] drafted: {customer_email[:80]}...")
handler.check_agent_response(customer_email, node="customer_email")

# ---------------------------------------------------------------
# THE MOMENT THAT MATTERS: does the email reach the customer?
# ---------------------------------------------------------------
print("\n" + "=" * 62)
print("  FINAL OUTPUT TO CUSTOMER")
print("=" * 62)
print(handler.get_safe_response(customer_email))

print("\n--- Detection summary ---")
for flag in handler.summary()["flags"]:
    print(f"  [{flag['severity'].upper()}] {flag['type']}: {flag['message']}")
