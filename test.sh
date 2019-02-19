#!/bin/bash
# 1. ./compile.sh
# 2. ./train.sh trainfile.json devfile.json model_file
# 3. ./test.sh model_file testfile.json outputfile.txt

#$1: model_file , $2: testfile.json, $3: outfutfile.txt

python3 test.py $1 $2 $3