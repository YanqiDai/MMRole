#!/bin/bash

python inference.py --model_path model_weights/MMRole-Agent --input_dir data/test/in-distribution --output_dir eval/in-test_answers/mmrole-agent
python inference.py --model_path model_weights/MMRole-Agent --input_dir data/test/out-of-distribution --output_dir eval/out-test_answers/mmrole-agent