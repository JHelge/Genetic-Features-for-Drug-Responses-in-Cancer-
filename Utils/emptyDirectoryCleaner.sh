#!/bin/bash

# Enable double asterisk globbing (see `man bash`).
shopt -s globstar

# Loop through every subdirectory.
for d in **/; do
    # Glob a list of the files in the directory.
    f=("$d")
    # If
    # 1. there is exactly 1 entry in the directory
    # 2. it is a file
    # 3. the file name consists of 10 digits with a ".jpg" suffix:
    if [[ ${#f[@]} -empty  ]]; then
        #echo` to ensure a test run; remove when verified.
        echo rm -r -- "$d"
    fi
done
