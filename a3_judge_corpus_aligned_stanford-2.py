#!/usr/bin/env python
# coding: utf-8
"""
Judge-specific corpus processing using EXACT Stanford methodology
Only difference: processes judge folders instead of single corpus
"""
import pandas as pd
import numpy as np
import re
import os
import tqdm
import nltk
import string
import pickle
from functions import *
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO)
                    
# scikit-learn bootstrap
from sklearn.utils import resample

os.makedirs("/u/home/i/iacir21/myscratch/clean_data/bootstrapped_judge_samples_train_vnorm", exist_ok=True)
os.makedirs("/u/home/i/iacir21/myscratch/clean_data/bootstrapped_judge_samples_test_vnorm", exist_ok=True)
os.makedirs("/u/home/i/iacir21/myscratch/clean_data/bootstrapped_judge_samples_eligible_vnorm", exist_ok=True)

# ---------------- CONFIG TRAIN ----------------
root_dir = "/u/home/i/iacir21/myscratch/replication/judges_corpus_train"
result_dir = "/u/home/i/iacir21/myscratch/clean_data/bootstrapped_judge_samples_train_vnorm"

# Load top 50k vocab
top_50k = open('/u/home/i/iacir21/myscratch/clean_data/Top_50k_words_100k_final', 'rb')
top_50k_words_list = pickle.load(top_50k)
stop_words = ["st", "nd", "th", "rd"]

# Process each judge folder
judge_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

for judge_name in judge_folders:
    judge_dir = os.path.join(root_dir, judge_name)
    judge_output_dir = os.path.join(result_dir, judge_name)
    os.makedirs(judge_output_dir, exist_ok=True)
    
    logging.info(f"Processing judge: {judge_name}")
    
    judgements = os.listdir(judge_dir)
    judgements.sort()
    sentences_list_doc_wise = []
    
    for i in tqdm.trange(len(judgements)):
        clean_sentences_list = []
        report_file = judgements[i]
        file_path = os.path.join(judge_dir, report_file)
        temp = ""
        
        # Try different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as fin:
                    temp = fin.read()
                logging.debug(f"Successfully read {report_file} with encoding: {encoding}")
                break  # Successfully read the file, exit the loop
            except UnicodeDecodeError:
                continue  # Try next encoding
            except Exception as e:
                logging.error(f"Error reading {report_file}: {e}")
                break
        else:
            # If all encodings failed, use utf-8 with errors='ignore' as fallback
            logging.warning(f"All encodings failed for {report_file}, using utf-8 with errors='ignore'")
            with open(file_path, "r", encoding='utf-8', errors='ignore') as fin:
                temp = fin.read()
        
        lines = temp.split(".")
        for line in lines:
            clean_line = cleaner(line)
            clean_sentences_list.append(clean_line)
        sentences_list_doc_wise.append(clean_sentences_list)
    
    logging.info("Data Cleaned, Starting Tokenising Now and keeping just tokens which are in top 50,000 most common tokens")
    
    sentences_tok_list_doc_wise = []
    for i in tqdm.trange(len(sentences_list_doc_wise)):
        sentences_tok_list = []
        for j in range(len(sentences_list_doc_wise[i])):
            tok_list = nltk.tokenize.word_tokenize(sentences_list_doc_wise[i][j])  #Tokenizing the cleaned sentences, and storing the tokens in a list
            new_tok_list = [word for word in tok_list if word in top_50k_words_list and word not in stop_words]  # taking only words which are in 50,000 most frequent words
            sentences_tok_list.append(new_tok_list)
        sentences_tok_list_doc_wise.append(sentences_tok_list)
    
    for index in range(len(sentences_tok_list_doc_wise)):
        opinion = sentences_tok_list_doc_wise[index]
        new_opinion = [sentence for sentence in opinion if sentence != []]  # removing empty lists
        sentences_tok_list_doc_wise[index] = new_opinion
    
    logging.info("Corpus Cleaned and Tokenised")
    
    for index in tqdm.trange(1, 26):
        bootstrap_docs = resample(sentences_tok_list_doc_wise, replace=True, n_samples=len(sentences_tok_list_doc_wise))
        bootstrap_sentences = []
        for doc in bootstrap_docs:
            for sent in doc:
                if sent:  # Only add non-empty sentences
                    bootstrap_sentences.append(sent)
        
        # Flatten the list of token lists
        flat_list = [token for sent in bootstrap_sentences for token in sent]
        corpus_final = " ".join(flat_list)
        
        with open(os.path.join(judge_output_dir, 'corpus_bstrap_sample_' + str(index)), 'w+', encoding='utf-8') as file:
            file.write(corpus_final)
            
        logging.info("Corpus saved for bootstrapped sample number " + str(index))
    
    logging.info(f"Cleaned corpora saved for all bootstrapped samples for judge {judge_name}")

logging.info("ALL TRAIN JUDGES PROCESSED")


#------------------------------------------------------------------------------------------------------------------    
#                  TEST
#------------------------------------------------------------------------------------------------------------------

# ---------------- CONFIG TEST ----------------
root_dir = "/u/home/i/iacir21/myscratch/replication/judges_corpus_test"
result_dir = "/u/home/i/iacir21/myscratch/clean_data/bootstrapped_judge_samples_test_vnorm"

# Process each judge folder
judge_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

for judge_name in judge_folders:
    judge_dir = os.path.join(root_dir, judge_name)
    judge_output_dir = os.path.join(result_dir, judge_name)
    os.makedirs(judge_output_dir, exist_ok=True)
    
    logging.info(f"Processing judge: {judge_name}")
    
    judgements = os.listdir(judge_dir)
    judgements.sort()
    sentences_list_doc_wise = []
    
    for i in tqdm.trange(len(judgements)):
        clean_sentences_list = []
        report_file = judgements[i]
        file_path = os.path.join(judge_dir, report_file)
        temp = ""
        
        # Try different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as fin:
                    temp = fin.read()
                logging.debug(f"Successfully read {report_file} with encoding: {encoding}")
                break  # Successfully read the file, exit the loop
            except UnicodeDecodeError:
                continue  # Try next encoding
            except Exception as e:
                logging.error(f"Error reading {report_file}: {e}")
                break
        else:
            # If all encodings failed, use utf-8 with errors='ignore' as fallback
            logging.warning(f"All encodings failed for {report_file}, using utf-8 with errors='ignore'")
            with open(file_path, "r", encoding='utf-8', errors='ignore') as fin:
                temp = fin.read()
        
        lines = temp.split(".")
        for line in lines:
            clean_line = cleaner(line)
            clean_sentences_list.append(clean_line)
        sentences_list_doc_wise.append(clean_sentences_list)
    
    logging.info("Data Cleaned, Starting Tokenising Now and keeping just tokens which are in top 50,000 most common tokens")
    
    sentences_tok_list_doc_wise = []
    for i in tqdm.trange(len(sentences_list_doc_wise)):
        sentences_tok_list = []
        for j in range(len(sentences_list_doc_wise[i])):
            tok_list = nltk.tokenize.word_tokenize(sentences_list_doc_wise[i][j])  #Tokenizing the cleaned sentences, and storing the tokens in a list
            new_tok_list = [word for word in tok_list if word in top_50k_words_list and word not in stop_words]  # taking only words which are in 50,000 most frequent words
            sentences_tok_list.append(new_tok_list)
        sentences_tok_list_doc_wise.append(sentences_tok_list)
    
    for index in range(len(sentences_tok_list_doc_wise)):
        opinion = sentences_tok_list_doc_wise[index]
        new_opinion = [sentence for sentence in opinion if sentence != []]  # removing empty lists
        sentences_tok_list_doc_wise[index] = new_opinion
    
    logging.info("Corpus Cleaned and Tokenised")
    
    for index in tqdm.trange(1, 26):
        bootstrap_docs = resample(sentences_tok_list_doc_wise, replace=True, n_samples=len(sentences_tok_list_doc_wise))
        bootstrap_sentences = []
        for doc in bootstrap_docs:
            for sent in doc:
                if sent:  # Only add non-empty sentences
                    bootstrap_sentences.append(sent)
        
        # Flatten the list of token lists
        flat_list = [token for sent in bootstrap_sentences for token in sent]
        corpus_final = " ".join(flat_list)
        
        with open(os.path.join(judge_output_dir, 'corpus_bstrap_sample_' + str(index)), 'w+', encoding='utf-8') as file:
            file.write(corpus_final)
            
        logging.info("Corpus saved for bootstrapped sample number " + str(index))
    
    logging.info(f"Cleaned corpora saved for all bootstrapped samples for judge {judge_name}")

logging.info("ALL TEST JUDGES PROCESSED")


#------------------------------------------------------------------------------------------------------
#                  ELIGIBLE
#------------------------------------------------------------------------------------------------------

# ---------------- CONFIG ELIGIBLE ----------------
root_dir = "/u/home/i/iacir21/myscratch/replication/judges_corpus_eligible"
result_dir = "/u/home/i/iacir21/myscratch/clean_data/bootstrapped_judge_samples_eligible_vnorm"

# Process each judge folder
judge_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

for judge_name in judge_folders:
    judge_dir = os.path.join(root_dir, judge_name)
    judge_output_dir = os.path.join(result_dir, judge_name)
    os.makedirs(judge_output_dir, exist_ok=True)
    
    logging.info(f"Processing judge: {judge_name}")
    
    judgements = os.listdir(judge_dir)
    judgements.sort()
    sentences_list_doc_wise = []
    
    for i in tqdm.trange(len(judgements)):
        clean_sentences_list = []
        report_file = judgements[i]
        file_path = os.path.join(judge_dir, report_file)
        temp = ""
        
        # Try different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as fin:
                    temp = fin.read()
                logging.debug(f"Successfully read {report_file} with encoding: {encoding}")
                break  # Successfully read the file, exit the loop
            except UnicodeDecodeError:
                continue  # Try next encoding
            except Exception as e:
                logging.error(f"Error reading {report_file}: {e}")
                break
        else:
            # If all encodings failed, use utf-8 with errors='ignore' as fallback
            logging.warning(f"All encodings failed for {report_file}, using utf-8 with errors='ignore'")
            with open(file_path, "r", encoding='utf-8', errors='ignore') as fin:
                temp = fin.read()
        
        lines = temp.split(".")
        for line in lines:
            clean_line = cleaner(line)
            clean_sentences_list.append(clean_line)
        sentences_list_doc_wise.append(clean_sentences_list)
    
    logging.info("Data Cleaned, Starting Tokenising Now and keeping just tokens which are in top 50,000 most common tokens")
    
    sentences_tok_list_doc_wise = []
    for i in tqdm.trange(len(sentences_list_doc_wise)):
        sentences_tok_list = []
        for j in range(len(sentences_list_doc_wise[i])):
            tok_list = nltk.tokenize.word_tokenize(sentences_list_doc_wise[i][j])  #Tokenizing the cleaned sentences, and storing the tokens in a list
            new_tok_list = [word for word in tok_list if word in top_50k_words_list and word not in stop_words]  # taking only words which are in 50,000 most frequent words
            sentences_tok_list.append(new_tok_list)
        sentences_tok_list_doc_wise.append(sentences_tok_list)
    
    for index in range(len(sentences_tok_list_doc_wise)):
        opinion = sentences_tok_list_doc_wise[index]
        new_opinion = [sentence for sentence in opinion if sentence != []]  # removing empty lists
        sentences_tok_list_doc_wise[index] = new_opinion
    
    logging.info("Corpus Cleaned and Tokenised")
    
    for index in tqdm.trange(1, 26):
        bootstrap_docs = resample(sentences_tok_list_doc_wise, replace=True, n_samples=len(sentences_tok_list_doc_wise))
        bootstrap_sentences = []
        for doc in bootstrap_docs:
            for sent in doc:
                if sent:  # Only add non-empty sentences
                    bootstrap_sentences.append(sent)
        
        # Flatten the list of token lists
        flat_list = [token for sent in bootstrap_sentences for token in sent]
        corpus_final = " ".join(flat_list)
        
        with open(os.path.join(judge_output_dir, 'corpus_bstrap_sample_' + str(index)), 'w+', encoding='utf-8') as file:
            file.write(corpus_final)
            
        logging.info("Corpus saved for bootstrapped sample number " + str(index))
    
    logging.info(f"Cleaned corpora saved for all bootstrapped samples for judge {judge_name}")

logging.info("ALL ELIGIBLE JUDGES PROCESSED")
