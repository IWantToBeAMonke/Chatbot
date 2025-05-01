class ContextManager:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def update(self, user_input, bot_response):
        self.history.append((user_input, bot_response))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        return "\n".join([f"User: {u}\nBot: {b}" for u, b in self.history])
