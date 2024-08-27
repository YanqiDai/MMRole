#!/bin/bash

MODEL="mmrole-agent"
python eval/RM_review.py --input_dir eval/mini_answers/$MODEL --output_dir eval/mini_reviews/$MODEL
python eval/RM_review.py --input_dir eval/zero_answers/$MODEL --output_dir eval/zero_reviews/$MODEL