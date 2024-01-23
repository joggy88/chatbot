import numpy as np
import random
import json

import torch
import torch.nn as ch
from torch.utils.data import Dataset, DataLoader

from nltk_Emily import bag_of_words, tokenize, stem
from model_Emily import NeuralNet

with open('queries_new.json', 'r') as file:
    queries = json.load(file)

all_new_words = []
title_new = []
ab = []
for query in queries['bot_queries']:
    title = query['title']
    # add to tag list
    title_new.append(title)
    for pattern in query['customer_input_queries']:
        # tokenize each word in the sentence
        w1 = tokenize(pattern)
        # add to our words list
        all_new_words.extend(w1)
        # add to ab pair
        ab.append((w1, title))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_new_words = [stem(w1) for w1 in all_new_words if w1 not in ignore_words]
# remove duplicates and sort
all_new_words = sorted(set(all_new_words))
title_new = sorted(set(title_new))

print(len(ab), "customer_input_queries")
print(len(title_new), "title:", title_new)
print(len(all_new_words), "unique stemmed words:", all_new_words)

# create training data
A_train = []
B_train = []
for (pattern_new_sentence, title) in ab:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_new_sentence, all_new_words)
    A_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = title_new.index(title)
    B_train.append(label)

A_train = np.array(A_train)
B_train = np.array(B_train)

# Hyper-parameters
num_epochs = 900
batch_new_size = 10
learning_new_rate = 0.002
input_new_size = len(A_train[0])
hidden_new_size = 10
output_new_size = len(title_new)
print(input_new_size, output_new_size)


class ChatDataset(Dataset):

    def __init__(self):
        self.c_samples = len(A_train)
        self.a_data = A_train
        self.b_data = B_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.a_data[index], self.b_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.c_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_new_size,
                          shuffle=True,
                          num_workers=0)

device_new = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1 = NeuralNet(input_new_size, hidden_new_size, output_new_size).to(device_new)

# Loss and optimizer
criterion = ch.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_new_rate)

# Train the model
for epoch in range(num_epochs):
    for (new_words, new_labels) in train_loader:
        new_words = new_words.to(device_new)
        new_labels = new_labels.to(dtype=torch.long).to(device_new)

        # Forward pass
        output = model1(new_words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss1 = criterion(output, new_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss1.item():.5f}')

print(f'final loss: {loss1.item():.5f}')

data1_new = {
    "model_new_state": model1.state_dict(),
    "input_new_size": input_new_size,
    "hidden_new_size": hidden_new_size,
    "output_new_size": output_new_size,
    "all_new_words": all_new_words,
    "title": title_new
}

FILE = "data1_new.pth"
torch.save(data1_new, FILE)

print(f'training complete. file saved to {FILE}')

