#!/bin/bash

MODEL="mmrole-agent"
python eval/RM_result.py --input_dir eval/mini_reviews/$MODEL --output_dir eval/mini_results/$MODEL
python eval/RM_result.py --input_dir eval/zero_reviews/$MODEL --output_dir eval/zero_results/$MODEL