from agent_debugger_live import LiveWatchHandler

handler = LiveWatchHandler()
handler.log_user_query("What is the current stock price of Apple?")
handler.check_agent_response("Apple's stock is currently trading at $182.50.")
