from agent_debugger_live import LiveWatchHandler

handler = LiveWatchHandler(use_semantic_layer=True)
handler.check_agent_response("The weather today is sunny with a high of 75 degrees, which is why I recommend bringing an umbrella.")
