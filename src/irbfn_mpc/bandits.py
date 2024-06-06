import numpy as np

def sigmoid(x):
    x = np.clip(x, -100, 100)
    return 1.0 / (1.0 + np.exp(-x))

class EXP3:
    def __init__(self, n, gamma) -> None:
        self.n = n
        self.weights = np.ones((n,))
        self.gamma = gamma
        self.sampling_brob = None

    def reset(self):
        self.weights = np.ones((self.n,))
        self.sampling_brob = None

    def pull_arm(self):
        self.sampling_prob = (1 - self.gamma) * (
            self.weights / np.sum(self.weights)
        ) + (self.gamma / self.n)
        pulled = np.random.choice(self.n, p=self.sampling_prob)
        return pulled

    def update_dist(self, i, r, rew_scale=0.5):
        r = sigmoid(rew_scale * r)
        rews = np.zeros((self.n,))
        rews[i] = r
        assert self.sampling_prob is not None, "Must pull arm before update"
        adj_rew = rews / self.sampling_prob
        self.weights = self.weights * np.exp(self.gamma * adj_rew / self.n)


def main():
    raise NotImplementedError("Not implemented.")

if __name__ == "__main__":
    main()