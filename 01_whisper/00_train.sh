#!/bin/sh

if [ $# = 0 ]; then
  # remove preprocess-data
  rm -rf dialect-preprocessed
  echo REMOVED dialect-preprocessed
  # make preprocess-data
  echo -------------------
  echo START PREPROCESSING
  echo -------------------
  python 10_preprocessing.py
  echo --------------------
  echo FINISH PREPROCESSING
  echo --------------------
elif [ $# = 1 ] && [ "$1" = "1" ]; then
  echo ------------------------
  echo no execute preprocessing
  echo ------------------------
fi

echo --------------------------------
echo START FINE-TUNING whisper-medium
echo --------------------------------
python 11_finetuning.py
