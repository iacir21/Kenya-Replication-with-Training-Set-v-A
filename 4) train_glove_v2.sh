#!/bin/bash
set -e

start_processing=false  # <---- initialize before loop

for judge_path in clean_data/bootstrapped_judge_samples/*; do
  judge=$(basename "$judge_path")
  
  if [ "$judge" = "abigail mshila" ]; then
      start_processing=true
  fi

  # Skip until the flag is true
  [ "$start_processing" = false ] && continue
  
  echo "Processing judge: $judge"

  for index in {1..25}; do
    echo "Making embeddings for judge $judge, sample $index"

    CORPUS="$judge_path/corpus_bstrap_sample_$index.txt"
   
    BUILDDIR="GloVe/build"
    SAVE_DIR="saved_embeddings/$judge"
    safe_judge=$(echo "$judge" | tr -cd '[:alnum:]_-')
    VOCAB_FILE="$SAVE_DIR/vocab_${safe_judge}_$index.txt"
    COOCCURRENCE_FILE="$SAVE_DIR/cooccurrence_${safe_judge}_$index.bin"
    COOCCURRENCE_SHUF_FILE="$SAVE_DIR/cooccurrence_${safe_judge}_$index.shuf.bin"
    SAVE_FILE="$SAVE_DIR/vectors_bstrap_sample_$index"
    VERBOSE=2
    MEMORY=4.0
    VOCAB_MIN_COUNT=5
    VECTOR_SIZE=300
    MAX_ITER=20
    WINDOW_SIZE=10
    BINARY=2
    NUM_THREADS=8
    X_MAX=10

    mkdir -p "$SAVE_DIR"

    if hash python 2>/dev/null; then
      PYTHON=python
    else
      PYTHON=python3
    fi

    echo
    echo "$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < "$CORPUS" > "$VOCAB_FILE""
    $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < "$CORPUS" > "$VOCAB_FILE"

    echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < "$CORPUS" > "$COOCCURRENCE_FILE""
    $BUILDDIR/cooccur -memory $MEMORY -vocab-file "$VOCAB_FILE" -verbose $VERBOSE -window-size $WINDOW_SIZE < "$CORPUS" > "$COOCCURRENCE_FILE"

    echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < "$COOCCURRENCE_FILE" > "$COOCCURRENCE_SHUF_FILE"

    echo "$ $BUILDDIR/glove -save-file \"$SAVE_FILE\" -threads $NUM_THREADS -input-file \"$COOCCURRENCE_SHUF_FILE\" -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file \"$VOCAB_FILE\" -verbose $VERBOSE"

    $BUILDDIR/glove -save-file "$SAVE_FILE" -threads $NUM_THREADS -input-file "$COOCCURRENCE_SHUF_FILE" \-x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file "$VOCAB_FILE" -verbose $VERBOSE

    if [ "$CORPUS" = 'corpus_temp' ]; then
      if [ "$1" = 'matlab' ]; then
        matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2
      elif [ "$1" = 'octave' ]; then
        octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
      else
        echo "$ $PYTHON eval/python/evaluate.py"
        $PYTHON eval/python/evaluate.py
      fi
    fi
  done
done

echo "Embeddings made for all judges and samples"
