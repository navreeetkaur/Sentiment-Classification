#!/bin/bash
# 1. ./compile.sh
# 2. ./train.sh trainfile.json devfile.json model_file
# 3. ./test.sh model_file testfile.json outputfile.txt

#$1: trainfile.json , $2: devfile.json, $3: model_file

mkdir data
python3 train.py $1 $2 $3