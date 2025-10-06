import os
import tqdm
import nltk
import pickle
import logging
from sklearn.utils import resample

from functions import cleaner  # your cleaner()

# ---------------- CONFIG ----------------
root_dir = "/u/home/i/iacir21/myscratch/replication/judges_corpus"
out_root = "/u/home/i/iacir21/myscratch/clean_data/bootstrapped_judge_samples"

# load top 50k vocab
with open("/u/home/i/iacir21/myscratch/clean_data/Top_50k_words_100k_final", "rb") as f:
    top_50k_words_list = pickle.load(f)

stop_words = ["st", "nd", "th", "rd"]

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

# to keep track of problematic judges
problematic_judges = []
# ---------------------------------------


def process_judge_folder(judge_name, judge_path):
    logging.info(f"Processing judge: {judge_name}")
    judgements = os.listdir(judge_path)
    judgements.sort()

    sentences_list_doc_wise = []
    judge_had_issue = False  # track per-judge issues

    # ---------- Step 1: Cleaning ----------
    for i in tqdm.trange(len(judgements), desc=f"Cleaning {judge_name}"):
        clean_sentences_list = []
        report_file = judgements[i]
        file_path = os.path.join(judge_path, report_file)

        try:
            try:
                with open(file_path, "r", encoding="utf-8") as fin:
                    temp = fin.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as fin:
                    temp = fin.read()

            lines = temp.split(".")
            for line in lines:
                clean_line = cleaner(line)
                if clean_line.strip():  # avoid adding empty lines
                    clean_sentences_list.append(clean_line)

        except Exception as e:
            logging.warning(f"Skipping file {report_file} for {judge_name} due to error: {e}")
            judge_had_issue = True
            continue

        # append only if we got something meaningful
        if clean_sentences_list:
            sentences_list_doc_wise.append(clean_sentences_list)

    if judge_had_issue:
        problematic_judges.append(judge_name)

    # ---------- Step 1 Diagnostics ----------
    total_clean_sentences = sum(len(doc) for doc in sentences_list_doc_wise)
    logging.info(f"{judge_name}: {total_clean_sentences} total cleaned sentences before tokenization")

    if total_clean_sentences == 0:
        logging.warning(f"No usable content for {judge_name} after cleaning. Skipping bootstrapping.")
        problematic_judges.append(judge_name)
        return

    logging.info("Data Cleaned, Starting Tokenising Now and keeping just tokens which are in top 50,000 most common tokens")

    # ---------- Step 2: Tokenisation ----------
    sentences_tok_list_doc_wise = []
    for i in tqdm.trange(len(sentences_list_doc_wise), desc=f"Tokenising {judge_name}"):
        sentences_tok_list = []
        for j in range(len(sentences_list_doc_wise[i])):
            tok_list = nltk.tokenize.word_tokenize(sentences_list_doc_wise[i][j])
            new_tok_list = [word for word in tok_list if word in top_50k_words_list and word not in stop_words]
            if new_tok_list:
                sentences_tok_list.append(new_tok_list)
        if sentences_tok_list:
            sentences_tok_list_doc_wise.append(sentences_tok_list)

    # ---------- Step 2 Diagnostics ----------
    total_tok_sentences = sum(len(doc) for doc in sentences_tok_list_doc_wise)
    total_tokens = sum(len(sent) for doc in sentences_tok_list_doc_wise for sent in doc)
    logging.info(f"{judge_name}: {total_tok_sentences} sentences and {total_tokens} tokens kept after filtering")

    if total_tok_sentences == 0 or total_tokens == 0:
        logging.warning(f"No usable tokens for {judge_name} after tokenization. Skipping bootstrapping.")
        problematic_judges.append(judge_name)
        return

    # ---------- Step 3: Remove empty sentences ----------
    for index in range(len(sentences_tok_list_doc_wise)):
        opinion = sentences_tok_list_doc_wise[index]
        new_opinion = [sentence for sentence in opinion if sentence != []]
        sentences_tok_list_doc_wise[index] = new_opinion

    # ---------- Step 4: Bootstrap and save ----------
    judge_out_dir = os.path.join(out_root, judge_name)
    os.makedirs(judge_out_dir, exist_ok=True)

    logging.info("Corpus Cleaned and Tokenised")
    for index in tqdm.trange(1, 26, desc=f"Bootstrapping {judge_name}"):
        bootstrap_docs = resample(
            sentences_tok_list_doc_wise, replace=True, n_samples=len(sentences_tok_list_doc_wise)
        )
        bootstrap_sentences = []
        for doc in bootstrap_docs:
            bootstrap_sentences.append("\n")
            for sent in doc:
                bootstrap_sentences.append(sent)

        bootstrap_sentences_new = [ele for ele in bootstrap_sentences if ele != []]
        flat_list = [item for sublist in bootstrap_sentences_new for item in sublist]
        corpus_final = " ".join(word for word in flat_list)

        out_file = os.path.join(judge_out_dir, f"corpus_bstrap_sample_{index}.txt")
        with open(out_file, "w+", encoding="utf-8") as fout:
            fout.write(corpus_final)

        logging.info(f"Corpus saved for judge {judge_name}, bootstrapped sample number {index}")

    logging.info(f"Finished judge {judge_name}. Files written to {judge_out_dir}")


# ---------- MAIN LOOP ----------
for judge_name in os.listdir(root_dir)[20:]:
    judge_path = os.path.join(root_dir, judge_name)
    if os.path.isdir(judge_path):
        process_judge_folder(judge_name, judge_path)

logging.info("All judges processed. Bootstrapped corpora saved.")

# ---------- Print out problematic judges ----------
if problematic_judges:
    logging.warning(f"The following judges had problematic or empty corpora: {problematic_judges}")
else:
    logging.info("No problematic judges encountered.")
