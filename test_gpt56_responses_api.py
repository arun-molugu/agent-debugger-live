"""Bounded experiment: gpt-5.6-terra via Responses API route."""
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from agent_debugger_live import LiveWatchHandler

llm = ChatOpenAI(model="gpt-5.6-terra", use_responses_api=True)

handler = LiveWatchHandler(verbose=True, block_on_critical=True)
agent = create_agent(model=llm, tools=[],
                     system_prompt="You are a helpful assistant.")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Say exactly: handler test OK"}]},
    config={"callbacks": [handler]},
)
print(result["messages"][-1].content)
print("SUCCESS - gpt-5.6-terra works via Responses API route")
