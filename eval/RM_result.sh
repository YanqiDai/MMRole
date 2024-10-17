#!/bin/bash

MODEL="mmrole-agent"
python eval/RM_result.py --input_dir eval/in-test_reviews/$MODEL --output_dir eval/in-test_results/$MODEL
python eval/RM_result.py --input_dir eval/out-test_reviews/$MODEL --output_dir eval/out-test_results/$MODEL