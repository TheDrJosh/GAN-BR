import numpy as np

LearningRate = 1


def activate(m):
    return 1 / (1 + np.exp(-m))


class NeuralNetwork:
    def __init__(self, shape):
        self.shape = shape
        self.weights = []
        self.biases = []
        for i in range(len(shape) - 1):
            self.weights.append(np.random.rand(self.shape[i + 1], self.shape[i]))
        for i in range(len(shape) - 1):
            self.biases.append(np.random.rand(self.shape[i + 1], 1))

    def run(self, inp):
        if len(inp) != self.shape[0]:
            exit(1)
        work_matrix = np.zeros((self.shape[0], 1))

        for i in range(len(self.shape) - 1):
            work_matrix = activate(self.weights[i] @ work_matrix + self.biases[i])

        return work_matrix

    def train(self, data, result):
        print('training')

    def decent(self, x, y):

        return 0

    def loss(self, data, result):
        l = 0
        for i in range(len(data)):
            l = l + np.power(self.run(data[i]) - result[i], 2)

        l = l * (1 / len(data))
        return l


training_data = [(0, 0), (0, 1), (1, 0), (1, 1)]
#        and bot,     or bot,      xor bot
expected = [[0, 0, 0, 1], [0, 1, 1, 1], [0, 1, 1, 0]]
bots = [
    NeuralNetwork([2, 3, 1]),
    NeuralNetwork([2, 3, 1]),
    NeuralNetwork([2, 3, 1])
]

before = [[], [], []]
after = [[], [], []]

for b in range(3):
    for d in range(4):
        before[b].append(bots[b].run(training_data[d]))
    bots[b].train(training_data[b], expected[b])
    for d in range(4):
        after[b].append(bots[b].run(training_data[d]))

for b in range(3):
    print("bot " + str(b + 1))
    for d in range(4):
        print("  " + str(training_data[d]) + ": " + str(before[b][d]) + " | " + str(after[b][d]))
    print()