from agent_debugger_live import LiveWatchHandler

class FakeOutput:
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return self.text

handler = LiveWatchHandler()
handler.on_tool_end(FakeOutput("Error: insufficient permissions"))
handler.check_agent_response("The retry succeeded and your request is complete.")
