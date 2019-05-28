class StatePacket:
    def __init__(self, trainer):
        self.trainer = trainer
        self.epsilon = []
        self.reward = []
        self.policy = []
        self.Q = []
        self.V = []
        self.A = []

    def update(self):
        self.reward.append(self.trainer.av_reward)
        self.epsilon.append(self.trainer.epsilon)
        self.Q.append(self.trainer.Q)
        self.V.append(self.trainer.V)
        self.A.append(self.trainer.A)
        self.policy.append(self.trainer.policy)
