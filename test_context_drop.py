from agent_debugger_live import LiveWatchHandler

handler = LiveWatchHandler()
handler.check_agent_response("Your account balance is 4500 rupees.")
handler.check_agent_response("As mentioned, your account balance is 3200 rupees.")

