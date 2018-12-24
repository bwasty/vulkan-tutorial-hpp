#!/bin/bash
stepFilename=$(basename $2)
echo $stepFilename
cp "$1" ./"$stepFilename"
git diff --color-words --no-index "$2" "$stepFilename"
git diff --no-index "$2" "$stepFilename" | tail -n+3 > "$stepFilename".diff
