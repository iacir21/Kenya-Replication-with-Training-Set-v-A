#!/usr/bin/env python
# coding: utf-8
import re
from glove import Corpus
from glove import Glove

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO)
                    
def cleaner(txt_file):
    
    text = txt_file
    text = re.sub(r"(&[a-zA-Z]*;)", " ", text)  # the txt files had some unwanted text like &rsquo; this line removes such text
    text = text.lower()
    
    # remove punctuation and numbers from the string
    punctuations = '''!()[]{};:'"\,<>./?@#$%^&*_~=+`|0123456789'''  # removing punctuations except hyphens
    
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, "")
            
    
    text = text.replace(" st ", " ")
    text = text.replace(" nd ", " ")
    text = text.replace(" rd ", " ")
    text = text.replace(" th ", " ")
    text = text.replace("hellip", " ")
    text = text.replace("rsquo", " ")
    text = text.replace("ldquo", " ")
    text = text.replace("rdquo", " ")
    text = text.replace("ndash", " ")
    text = text.replace("--", " ")

    
    return text

#training from here
def train_embeddings(bootstrap_sentences_new, sample_num):

    corpus_model = Corpus()
    corpus_model.fit(bootstrap_sentences_new, window=10)

    epochs = 20
    no_threads = 10

    glove = Glove(no_components = 300, learning_rate=0.05)
    glove.fit(corpus_model.matrix,epochs=epochs,no_threads=no_threads,verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    glove.save('saved_embeddings/glove_bootstrap_full_corpus_100k_sample_' + str(sample_num) + '_glove.gl')
    
