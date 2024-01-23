import torch
import torch.nn as ch


class NeuralNet(ch.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = ch.Linear(input_size, hidden_size)
        self.l2 = ch.Linear(hidden_size, hidden_size)
        self.l3 = ch.Linear(hidden_size, num_classes)
        self.relu =ch.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out