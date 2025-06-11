import random
import torch
import json
import os
from nltk_utils import bag_of_words, tokenize
from model import NeuralNet

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'intents.json'), 'r', encoding='utf-8') as f:
    intents = json.load(f)

data = torch.load(os.path.join(script_dir, 'data.pth'), map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

confidence_threshold = 0.75
fallback_responses = [
    "Maaf, aku belum mengerti pertanyaanmu.",
    "Coba tanyakan dengan cara lain ya.",
]

def get_response_from_model(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted_idx = torch.max(output, dim=1)
    tag = tags[predicted_idx.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted_idx.item()]

    if prob.item() > confidence_threshold:
        for intent in intents['intents']:
            if intent["tag"] == tag:
                return random.choice(intent['responses'])
    return random.choice(fallback_responses)

if __name__ == "__main__":
    print("MaydarlingBot aktif! Ketik 'keluar' untuk mengakhiri.\n")
    while True:
        sentence = input("Kamu: ")
        if sentence.lower() in ["keluar", "quit", "exit"]:
            print("MaydarlingBot: Sampai jumpa! 💚")
            break

        response = get_response_from_model(sentence)
        print(f"MaydarlingBot: {response}")
