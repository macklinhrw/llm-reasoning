#!/bin/bash

./configure.sh
python ./qwen1.5-finetune.py "$@"
