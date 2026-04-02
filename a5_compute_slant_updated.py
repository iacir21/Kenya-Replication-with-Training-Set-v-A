import os
import json
import math
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# --- Configuration ---
EMBEDDINGS_ROOT = "/u/home/i/iacir21/myscratch/saved_embeddings_eligible"
NAME_MAP_FILE   = os.path.join(EMBEDDINGS_ROOT, "judge_name_map_eligible.json")
N_SAMPLES       = 25

OUTPUT_CSV = "/u/home/i/iacir21/myscratch/replication/judge_slant_scores_eligible.csv"

# --- Word sets (slant methodology) ---
male_words   = ["his", "he", "him", "mr", "himself"]
female_words = ["her", "she", "ms", "mrs", "herself"]
good_words   = ["competent", "strong", "power", "serious", "professional"]
bad_words    = ["frivolous", "vain", "incompetent", "unreasonable", "incapable"]

# --- Helper functions (from validation notebook) ---

def cosine_similarity(x, y):
    """Manual cosine similarity."""
    s = norm_x2 = norm_y2 = 0.0
    for xi, yi in zip(x, y):
        s       += xi * yi
        norm_x2 += xi ** 2
        norm_y2 += yi ** 2
    denom = math.sqrt(norm_x2 * norm_y2)
    return s / denom if denom > 0 else np.nan


def load_glove_model(path):
    """Load a GLoVe .txt file into a {word: np.ndarray} dict.
    Handles UTF-8 and Latin-1 encodings."""
    model = {}
    for enc in ('utf-8', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc) as f:
                for line in f:
                    parts = line.rstrip().split()
                    if len(parts) < 2:
                        continue
                    word = parts[0]
                    vec  = np.array([float(v) for v in parts[1:]], dtype=np.float32)
                    model[word] = vec
            return model
        except UnicodeDecodeError:
            continue
    raise IOError(f"Could not decode {path} with utf-8 or latin-1.")


def find_embedding_file(judge_dir, sample_idx):
    """Return the .txt embedding path for a given judge dir and sample index."""
    candidate = os.path.join(judge_dir, f"vectors_bstrap_sample_{sample_idx}.txt")
    return candidate if os.path.isfile(candidate) else None


def mean_vector(words, model):
    """Mean embedding vector for a list of words (skips missing)."""
    vecs = [model[w] for w in words if w in model]
    return np.mean(vecs, axis=0) if vecs else None


def compute_slant_for_sample(model, male_words, female_words, good_words, bad_words):
    """
    Compute the gender-competence slant score for one bootstrap sample.

    gender_vec  = mean(male_words)  - mean(female_words)
    goodbad_vec = mean(good_words)  - mean(bad_words)
    slant       = cosine(gender_vec, goodbad_vec)

    Returns float (cosine similarity) or np.nan if any word group is absent.
    """
    gender_vec  = mean_vector(male_words,   model)
    anti_gender = mean_vector(female_words, model)
    goodbad_vec = mean_vector(good_words,   model)
    anti_good   = mean_vector(bad_words,    model)

    if any(v is None for v in [gender_vec, anti_gender, goodbad_vec, anti_good]):
        return np.nan

    gender_dim  = gender_vec  - anti_gender
    goodbad_dim = goodbad_vec - anti_good

    return cosine_similarity(gender_dim, goodbad_dim)


# --- Load judge name map ---
if os.path.isfile(NAME_MAP_FILE):
    with open(NAME_MAP_FILE) as f:
        judge_name_map = json.load(f)
    print(f"Loaded judge name map: {len(judge_name_map)} judges from {NAME_MAP_FILE}")
else:
    print(f"WARNING: {NAME_MAP_FILE} not found – building map from folder names.")
    judge_name_map = {}
    for d in sorted(os.listdir(EMBEDDINGS_ROOT)):
        full = os.path.join(EMBEDDINGS_ROOT, d)
        if os.path.isdir(full) and not d.startswith('.'):
            judge_name_map[d] = d.replace('_', ' ')

# Discover all judge folders on disk
judge_folders = sorted([
    d for d in os.listdir(EMBEDDINGS_ROOT)
    if os.path.isdir(os.path.join(EMBEDDINGS_ROOT, d)) and not d.startswith('.')
])
print(f"Found {len(judge_folders)} judge folders in {EMBEDDINGS_ROOT}")

# --- Main computation ---
rows = []

for judge_safe in tqdm(judge_folders, desc="Judges"):
    judge_dir      = os.path.join(EMBEDDINGS_ROOT, judge_safe)
    judge_original = judge_name_map.get(judge_safe, judge_safe.replace('_', ' '))

    slant_scores = {}   # {sample_idx: float}
    n_found      = 0

    for k in range(1, N_SAMPLES + 1):
        emb_path = find_embedding_file(judge_dir, k)
        if emb_path is None:
            print(f"  ⚠ Missing embedding for {judge_safe}: sample {k}")
            continue
        n_found += 1

        try:
            model = load_glove_model(emb_path)
        except Exception as e:
            print(f"  ⚠ Could not load {emb_path}: {e}")
            continue

        score = compute_slant_for_sample(model, male_words, female_words, good_words, bad_words)

        if np.isnan(score):
            print(f"  ⚠ NaN slant for {judge_safe} sample {k} — word group(s) missing from vocab")

        slant_scores[k] = score

    # Build row — one column per sample, plus summary stats
    row = {
        'judge_safe':      judge_safe,
        'judge_name':      judge_original,
        'n_samples_found': n_found,
    }

    for k in range(1, N_SAMPLES + 1):
        row[f'slant_goodbad_{k}'] = slant_scores.get(k, np.nan)

    valid_scores = [v for v in slant_scores.values() if not np.isnan(v)]
    row['slant_mean']   = np.mean(valid_scores)   if valid_scores else np.nan
    row['slant_median'] = np.median(valid_scores) if valid_scores else np.nan
    row['slant_std']    = np.std(valid_scores)    if valid_scores else np.nan

    rows.append(row)

# --- Save results ---
slant_df = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
slant_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Slant scores computed for {len(slant_df)} judges.")
print(f"   Saved → {OUTPUT_CSV}")
