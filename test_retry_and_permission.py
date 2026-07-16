from agent_debugger_live import LiveWatchHandler

class FakeOutput:
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return self.text

handler = LiveWatchHandler()
handler.on_tool_end(FakeOutput("Error: insufficient permissions to access account"))
handler.check_agent_response("Retrying the request now.")
handler.check_agent_response("Attempting again to process this.")
