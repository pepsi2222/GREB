from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained(
                '/data1/home/xingmei/GRE/e5-base-v2',
                use_fast=True,
                revision='main',
                trust_remote_code=True,
            )

len_of_rw = 0
with open('review.json', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        len_of_rw += len(tokenizer(data['text'])['input_ids'])
print(len_of_rw, len_of_rw / 6990280)
# 927649311 132.70560134930219