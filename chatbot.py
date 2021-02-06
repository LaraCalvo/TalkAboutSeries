import random
import json

import torch

from model import NeuralNet
from text_preprocess import bag_of_words, tokenize

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

########################
#Chatbot
#Train the chatbot first
########################

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

#Retrieve training data that was saved
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


#Parameters
prob_min = 0.6 #Minimum probability for choosing response
bot_name = "Sam" #Choose the chatbot name


########################
#Graphical interface
########################

"""
    Instructions for graphical interface
    - Submit text with submit button or by pressing enter
    - Quit by writting 'quit', 'bye', 'exit'. The bot will quit automatically after receiving a message indicating leave
    - Your conversations are not saved
"""


from tkinter import *
import tkinter as tk

root = Tk()
root.title("Talk about series")
root.geometry("600x400")

#Allow enter to also submit
root.bind_class("Button", "<Key-Return>", lambda event: event.widget.invoke())

title = Label(root, text="Talk about series chatbot", fg='red', font=('Helvetica', 16))
title.grid(row=0, column=0, pady=10)

#Conversation display
msg = Text(root, height = 15, width = 52)
msg.grid(row=1, column=0, padx=(50, 20))

#Scrollbar
scroll_bar = Scrollbar(root)
scroll_bar.grid(row=1, column=1)

sentence_var = tk.StringVar()

#Two methods: button or enter button
def submit():
    sentence = sentence_var.get()
    sentence_var.set("")
    msg.insert(tk.END, "You: " + sentence + "\n")
    sentence = tokenize(sentence)
    
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1) #Get the softmax activation values
    prob = probs[0][predicted.item()] #Prediction value for tag
    if prob.item() > prob_min:
        for intent in intents['intents']:
            if tag == intent["tag"]:    
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
                msg.insert(tk.END, "Bot: " + random.choice(intent['responses']) +"\n")
                if "exit" == tag:
                    quit()
    else:
        msg.insert(tk.END, bot_name + ": I don't understand\n")

def submit_e(e):
    sentence = sentence_var.get()
    sentence_var.set("")
    msg.insert(tk.END, "You: " + sentence + "\n")
    sentence = tokenize(sentence)
    
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1) #Get the softmax activation values
    prob = probs[0][predicted.item()] #Prediction value for tag
    if prob.item() > prob_min:
        for intent in intents['intents']:
            if tag == intent["tag"]:    
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
                msg.insert(tk.END, "Bot: " + random.choice(intent['responses']) +"\n")
                if "exit" == tag:
                    quit()
    else:
        msg.insert(tk.END, bot_name + ": I don't understand...\n")
    
    

#Button for submit
submit_btn = tk.Button(root, text = "Submit", command = submit)
submit_btn.grid(row=2, column=1)

#Enter text for chatbot
sentence_entry = tk.Entry(root,
                          textvariable = sentence_var,
                          font=('calibre', 10, 'normal'))
sentence_entry.grid(row=2, column=0, pady=10)

sentence_entry.bind('<Return>', submit_e)

root.mainloop()
