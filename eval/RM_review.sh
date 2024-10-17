#!/bin/bash

MODEL="mmrole-agent"
python eval/RM_review.py --input_dir eval/in-test_answers/$MODEL --output_dir eval/in-test_reviews/$MODEL
python eval/RM_review.py --input_dir eval/out-test_answers/$MODEL --output_dir eval/out-test_reviews/$MODEL