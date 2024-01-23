import random
import json

import torch

from model_Emily import NeuralNet
from nltk_Emily import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('queries_new.json', 'r') as json_data:
    queries_new = json.load(json_data)

FILE = "data1.pth"
data1 = torch.load(FILE)

input_new_size = data1["input_new_size"]
hidden_new_size = data1["hidden_new_size"]
output_new_size = data1["output_new_size"]
all_new_words = data1['all_new_words']
title_new = data1['title']
model_new_state = data1["model_new_state"]

model = NeuralNet(input_new_size, hidden_new_size, output_new_size).to(device)
model.load_state_dict(model_new_state)
model.eval()

bot_name = "Emily"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_new_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    title = title_new[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for query in queries_new['bot_queries']:
            if title == query["title"]:
                return random.choice(query['bot_responses'])

    return "I do not understand..."











