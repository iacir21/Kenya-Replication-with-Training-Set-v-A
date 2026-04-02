#!/bin/bash
set -e

# Trains GloVe embeddings for judge-specific bootstrapped samples.
# Iterates over each judge folder in clean_data/bootstrapped_judge_samples_eligible/
# and trains 25 bootstrap samples per judge.
# Output is saved under saved_embeddings/<judge_name_underscored>/
#
# Before training starts, exports a JSON map of safe folder names back to
# original judge names -> saved_embeddings/judge_name_map.json
# Use this file to merge embeddings with your metadata in Python.
#
# Usage:
#   bash train_embeddings_judgewise.sh
#   bash train_embeddings_judgewise.sh --start-from "Judge Name"
#
# --start-from "Judge Name"  : Skip all judges that come before the given name
#                              alphabetically. Useful for resuming after a failure.
#                              The name must match the folder name exactly
#                              (spaces are fine, casing must match).

# --- Parse arguments ---
START_FROM="rose edwina atieno ougo"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash $0 [--start-from \"Judge Name\"]"
            exit 1
            ;;
    esac
done

if [[ -n "$START_FROM" ]]; then
    echo "Resuming from judge: \"$START_FROM\" (all earlier judges will be skipped)"
    SKIPPING=true
else
    SKIPPING=false
fi

# --- GloVe parameters ---
JUDGE_SAMPLES_DIR="clean_data/bootstrapped_judge_samples_eligible_vnorm"
BUILDDIR="glove/build"
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=20
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=8
X_MAX=10

if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

# --- Export judge name map to JSON ---
# Writes saved_embeddings/judge_name_map.json with entries:
#   { "safe_folder_name": "Original Judge Name", ... }
# This is done BEFORE training so the map is always available even if
# training fails or is resumed with --start-from.
mkdir -p saved_embeddings_eligible
NAME_MAP_FILE="saved_embeddings_eligible/judge_name_map_eligible.json"
echo "Building judge name map -> $NAME_MAP_FILE"
echo "{" > "$NAME_MAP_FILE"
FIRST_ENTRY=true
while IFS= read -r -d '' JUDGE_DIR; do
    JUDGE_NAME=$(basename "$JUDGE_DIR")
    JUDGE_SAFE=$(echo "$JUDGE_NAME" | tr ' ' '_')
    if [[ "$FIRST_ENTRY" == true ]]; then
        FIRST_ENTRY=false
    else
        echo "," >> "$NAME_MAP_FILE"
    fi
    # Escape any double quotes in judge names just in case
    JUDGE_NAME_ESCAPED=$(echo "$JUDGE_NAME" | sed 's/"/\\"/g')
    JUDGE_SAFE_ESCAPED=$(echo "$JUDGE_SAFE" | sed 's/"/\\"/g')
    printf '  "%s": "%s"' "$JUDGE_SAFE_ESCAPED" "$JUDGE_NAME_ESCAPED" >> "$NAME_MAP_FILE"
done < <(find "$JUDGE_SAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d -not -name '.*' -print0 | sort -z)
echo "" >> "$NAME_MAP_FILE"
echo "}" >> "$NAME_MAP_FILE"
echo "Judge name map saved to $NAME_MAP_FILE"
echo ""

# --- Main training loop ---
# Iterate over each judge folder; using find + while loop to safely handle
# folder names that contain spaces. -not -name '.*' excludes hidden folders
# like .ipynb_checkpoints. Results are sorted so --start-from is predictable.
while IFS= read -r -d '' JUDGE_DIR; do

    # Extract the raw judge name from the folder path
    JUDGE_NAME=$(basename "$JUDGE_DIR")

    # --- Start-from logic: skip judges until we reach the target ---
    if [[ "$SKIPPING" == true ]]; then
        if [[ "$JUDGE_NAME" == "$START_FROM" ]]; then
            SKIPPING=false
            echo "Found start judge: \"$JUDGE_NAME\" — resuming now."
        else
            echo "Skipping judge: $JUDGE_NAME"
            continue
        fi
    fi

    # Create a filesystem-safe version of the judge name (spaces -> underscores)
    # This is the name used for output folders so you can merge with metadata.
    JUDGE_SAFE=$(echo "$JUDGE_NAME" | tr ' ' '_')

    echo "========================================================"
    echo "Processing judge: $JUDGE_NAME  (saved as: $JUDGE_SAFE)"
    echo "========================================================"

    # Create judge-specific output directory
    JUDGE_SAVE_DIR="saved_embeddings_eligible/$JUDGE_SAFE"
    mkdir -p "$JUDGE_SAVE_DIR"

    # Write a single-line name map into the judge's own folder immediately
    # so it is recoverable even if the script breaks before finishing.
    echo "{\"$JUDGE_SAFE\": \"$JUDGE_NAME\"}" > "$JUDGE_SAVE_DIR/judge_name_map.json"

    for index in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do

        echo ""
        echo "  Making embeddings for $JUDGE_NAME — sample $index"

        # Input corpus for this judge + bootstrap sample
        CORPUS="$JUDGE_DIR/corpus_bstrap_sample_$index"

        # Intermediate files are judge-scoped to avoid collisions across judges
        VOCAB_FILE="$JUDGE_SAVE_DIR/vocab_sample_${index}.txt"
        COOCCURRENCE_FILE="$JUDGE_SAVE_DIR/cooccurrence_sample_${index}.bin"
        COOCCURRENCE_SHUF_FILE="$JUDGE_SAVE_DIR/cooccurrence_sample_${index}.shuf.bin"

        # Output embedding file, named to match judge + sample for easy merging
        SAVE_FILE="$JUDGE_SAVE_DIR/vectors_bstrap_sample_$index"

        echo "  $ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < \"$CORPUS\" > \"$VOCAB_FILE\""
        $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < "$CORPUS" > "$VOCAB_FILE"

        echo "  $ $BUILDDIR/cooccur -memory $MEMORY -vocab-file \"$VOCAB_FILE\" -verbose $VERBOSE -window-size $WINDOW_SIZE < \"$CORPUS\" > \"$COOCCURRENCE_FILE\""
        $BUILDDIR/cooccur -memory $MEMORY -vocab-file "$VOCAB_FILE" -verbose $VERBOSE -window-size $WINDOW_SIZE < "$CORPUS" > "$COOCCURRENCE_FILE"

        echo "  $ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < \"$COOCCURRENCE_FILE\" > \"$COOCCURRENCE_SHUF_FILE\""
        $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < "$COOCCURRENCE_FILE" > "$COOCCURRENCE_SHUF_FILE"

        echo "  $ $BUILDDIR/glove -save-file \"$SAVE_FILE\" -threads $NUM_THREADS -input-file \"$COOCCURRENCE_SHUF_FILE\" -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file \"$VOCAB_FILE\" -verbose $VERBOSE"
        $BUILDDIR/glove \
            -save-file "$SAVE_FILE" \
            -threads $NUM_THREADS \
            -input-file "$COOCCURRENCE_SHUF_FILE" \
            -x-max $X_MAX \
            -iter $MAX_ITER \
            -vector-size $VECTOR_SIZE \
            -binary $BINARY \
            -vocab-file "$VOCAB_FILE" \
            -verbose $VERBOSE

    done

    echo "  Done with all samples for judge: $JUDGE_NAME"

done < <(find "$JUDGE_SAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d -not -name '.*' -print0 | sort -z)

# Warn if the requested start judge was never found in the directory listing
if [[ "$SKIPPING" == true ]]; then
    echo ""
    echo "WARNING: --start-from judge \"$START_FROM\" was never found in $JUDGE_SAMPLES_DIR"
    echo "Check that the name matches the folder name exactly (casing and spaces)."
    exit 1
fi

echo ""
echo "========================================================"
echo "Embeddings made for all judges and all samples."
echo "Judge name map available at: saved_embeddings_eligible/judge_name_map.json"
echo "========================================================"
