from tqdm import tqdm
from llama import Tokenizer
from llama.tokenizer import ChatFormat
import json
import sys

max_seq_len = 4096

if __name__ == '__main__':
    category = sys.argv[1]

    tokenizer = Tokenizer(model_path='/data1/home/xingmei/llama3-demo/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model')
    formatter = ChatFormat(tokenizer)

    review = json.load(open(f'/data1/home/xingmei/GRE/data/GoogleLocalData/Food/extracted_review/{category}.json'))
                
    dialogs = []
    gmap_ids = []
    for gmap_id, texts in tqdm(review.items()):

        dialog = [
            # {
            #     "role": "system",
            #     "content": "You are an excellent review summarizer.",
            # },
            {   "role": "user", 
                "content": """Given the user reviews of a restaurant, craft a concise summary that captures the restaurant's characteristics. Please give the proportion of reviews for each characteristic.

[User Reviews]
"""         },
        ] 
        token_left = max_seq_len - len(formatter.encode_dialog_prompt(dialog))

        gmap_ids.append(gmap_id)
        prompt = ''
        for i, text in enumerate(texts):
            token_len = len(tokenizer.encode(text, bos=True, eos=True))
            token_left -= token_len
            if token_left < 0:
                break
            prompt += f"{i + 1}. {text}\n"
        dialog[-1]['content'] += prompt
        assert len(formatter.encode_dialog_prompt(dialog)) <= max_seq_len + 100, len(formatter.encode_dialog_prompt(dialog))

        dialogs.append(dialog)

    with open(f'/data1/home/xingmei/GRE/data/GoogleLocalData/Food/extracted_review/prompt_{category}.json', 'w') as f:
        json.dump([dialogs, gmap_ids], f)