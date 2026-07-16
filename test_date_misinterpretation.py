from agent_debugger_live import LiveWatchHandler

handler = LiveWatchHandler()

class FakeOutput:
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return self.text

handler.on_tool_end(FakeOutput("Meeting scheduled_for 2026-03-15"))
print("last_scheduled_date after tool call:", handler.last_scheduled_date)

handler.check_agent_response("Your meeting is confirmed for March 12th at 10am.")

