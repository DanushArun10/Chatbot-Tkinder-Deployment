
import random
import json

import torch

from model import NeuralNet
from train import bag_of_words,tokenize

device = torch.device('cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    v, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)

    prob = probs[0][predicted.item()]
    #print(prob)
    #print(probs)
    print(prob)
    if prob.item() > 0.75:
        #print(prob.item)
        for intent in intents['intents']:
            if tag == intent["tag"]:
                k = (random.choice(intent['responses']))
                print(f"{bot_name}:", k)
                #time.sleep(5)
                return k

    return "I do not understand..."

