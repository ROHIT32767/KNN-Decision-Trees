#!/bin/bash
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_input_file>"
    exit 1
fi

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

python3 KNN.py "$input_file"
