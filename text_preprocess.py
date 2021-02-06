import numpy as np
import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

###########################
#Text preprocessing methods
###########################

def noise_removal(words):
    noise = ['?', '!', '.', ',', '[', ']', '-', '_']
    words = [word for word in words if word not in noise]
    return words
        

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())

#Filter stop words and duplicates
def filter_stop_words(words):
    stop_words = set(stopwords.words('english'))
    words_f = []
    for word in words:
        if word not in stop_words:
            words_f.append(word)
    words = sorted(set(words_f))
    return words


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    #The bag starts as an array of zeros
    bag = np.zeros(len(words), dtype=np.float32)
    for index, word in enumerate(words):
        if word in sentence_words: 
            bag[index] = 1
            #We put a 1 in the position of the word
    return bag


def preprocess(words):
    words = noise_removal(words)
    words = [stem(w) for w in words]
    words = filter_stop_words(words)
    return words
