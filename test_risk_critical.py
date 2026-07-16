from agent_debugger_live import LiveWatchHandler

handler = LiveWatchHandler()
handler.check_agent_response("Warning: unusual activity detected on this account.")
handler.check_agent_response("Critical: database connection lost during transaction.")

