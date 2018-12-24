#!/bin/bash
cp "$1" ./"$(basename $2)"
git diff --color-words --no-index "$1" "$2"
git diff --no-index "$1" "$2" | tail -n+3 > "$(basename $2)".diff
