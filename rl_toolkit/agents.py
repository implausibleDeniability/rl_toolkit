class Agent:
    """Base class for agents"""

    backbone = None

    def __init__(self, observation_space, action_space):
        self.action_space = action_space

    def sample_action():
        raise NotImplementedError("Abstract class Agent is used")

    def get_best_action():
        raise NotImplementedError("Abstract class Agent is used")

    def eval(
        self
    ):
        self.backbone.eval()

    def train(
        self
    ):
        self.backbone.train()

