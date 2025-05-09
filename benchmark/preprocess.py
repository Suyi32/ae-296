import sys
import json
from tqdm import tqdm
from transformers import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
word_tokenize = lambda x: tokenizer(x).input_ids
with open(sys.argv[1], 'r') as f:
    content = json.load(f)

with open(sys.argv[2], 'w') as of:
    out_content = []
    for line in tqdm(content):
        line['instruction_tokens'] = len(word_tokenize(line['instruction']))
        line['input_tokens'] = len(word_tokenize(line['input']))
        line['output_tokens'] = len(word_tokenize(line['output']))
        out_content.append(line)
    json.dump(out_content, of)
