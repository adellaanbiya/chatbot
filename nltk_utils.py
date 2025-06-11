import numpy as np
import nltk
#nltk.download('punkt') # Jalankan ini sekali jika belum, lalu komentari lagi

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer Bahasa Indonesia (Sastrawi)
factory = StemmerFactory()
stemmer_sastrawi = factory.create_stemmer()

def tokenize(sentence):
    """
    memisahkan kalimat menjadi array
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word using Sastrawi for Indonesian
    """
    return stemmer_sastrawi.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array
    """
    # stem each word in the input sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word in the vocabulary
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag