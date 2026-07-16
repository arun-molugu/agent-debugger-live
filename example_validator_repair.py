import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

from agent_debugger_live import LiveWatchHandler

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='your-key-here'")

handler = LiveWatchHandler()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class LoopState(TypedDict):
    question: str
    validator_feedback: str
    repair_output: str
    round: int


def validator_node(state: LoopState) -> LoopState:
    prompt = f"""You are a strict math question validator.
Question: {state['repair_output'] or state['question']}
Find ONE specific notation or formatting issue with this question, even if it's minor.
Respond in one short sentence describing the issue."""
    response = llm.invoke(prompt)
    feedback = response.content
    handler.check_agent_response(feedback, is_validator_complaint=True)
    print(f"[Round {state['round']}] VALIDATOR says: {feedback}")
    return {**state, "validator_feedback": feedback, "round": state["round"] + 1}


def repair_node(state: LoopState) -> LoopState:
    prompt = f"""You are fixing a math question based on this feedback: {state['validator_feedback']}
Original question: {state['question']}
Rewrite the question to fix the issue. Keep it brief."""
    response = llm.invoke(prompt)
    repaired = response.content
    handler.check_agent_response(repaired)
    print(f"[Round {state['round']}] REPAIR produces: {repaired[:100]}...")
    return {**state, "repair_output": repaired}


def should_continue(state: LoopState) -> str:
    if state["round"] >= 6:
        return END
    return "validator"


graph = StateGraph(LoopState)
graph.add_node("validator", validator_node)
graph.add_node("repair", repair_node)
graph.set_entry_point("validator")
graph.add_edge("validator", "repair")
graph.add_conditional_edges("repair", should_continue, {"validator": "validator", END: END})

app = graph.compile()


if __name__ == "__main__":
    initial_state = {
        "question": "Find the value of u_n where u_(n+1) = 2u_n + 1",
        "validator_feedback": "",
        "repair_output": "",
        "round": 1
    }
    print("=== Running validator/repair loop (max 6 rounds) ===\n")
    final_state = app.invoke(initial_state)

    print("\n=== Loop finished ===")
    print(f"Total rounds: {final_state['round'] - 1}")

    print("\n=== Detection summary ===")
    summary = handler.summary()
    print(f"Total steps watched: {summary['total_steps']}")
    print(f"Total flags raised: {summary['total_flags']}")
    for flag in summary["flags"]:
        print(f"  - {flag['type']}: {flag['message']}")
