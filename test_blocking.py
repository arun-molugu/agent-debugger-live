from agent_debugger_live import LiveWatchHandler

class FakeOutput:
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return self.text

handler = LiveWatchHandler(block_on_critical=True)
handler.on_tool_end(FakeOutput(""))
handler.check_agent_response("Your refund has been processed successfully.")

if handler.should_block():
    print(">>> This response would NOT be sent to the user.")
else:
    print(">>> Response sent normally.")
