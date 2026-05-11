class MockOut:
    def __init__(self, t): self.text = t
class MockReq:
    def __init__(self, id, t):
        self.request_id = id
        self.outputs = [MockOut(t)]

# If we had step_outputs ...
