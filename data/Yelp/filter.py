import os
import pandas as pd
import pickle
import re
import sys
import numpy as np
from tqdm import tqdm
work_dir = re.search('(.*GRE).*', os.getcwd(), re.IGNORECASE).group(1)
sys.path.append(work_dir)
sys.path.append(os.path.join(work_dir, 'RecStudio'))
from RecStudio.recstudio.data import SeqDataset
from RecStudio.recstudio.data.dataset import TensorFrame
from RecStudio.recstudio.utils import parser_yaml
from RecStudio.recstudio.utils import *
from collections import defaultdict
import random
import json
import csv
csv.field_size_limit(sys.maxsize)

from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import set_seed
set_seed(42)

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False
    
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_garbage(text):
    garbage_pattern = re.compile(r'\\x[a-fA-F0-9]{2}')
    return garbage_pattern.sub(r'', text)

def remove_html_tags(text):
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def remove_urls(text):
    url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_emails(text):
    email_pattern = re.compile(r'\S+@\S+')
    return email_pattern.sub(r'', text)

def remove_extra_whitespace(text):
    whitespace_pattern = re.compile(r'\s+')
    return whitespace_pattern.sub(r' ', text).strip()

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def remove_similar_sentences_tfidf(text, threshold=0.7):
    # 将文本拆分为句子
    sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    
    # 使用TF-IDF向量化
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    
    # 计算余弦相似度矩阵
    cosine_matrix = cosine_similarity(vectors)
    
    # 获取上三角矩阵的索引，并设置对角线为0以避免自相似
    np.fill_diagonal(cosine_matrix, 0)
    
    # 检查每个句子与其他句子的相似度，超过阈值则标记为相似
    similar_sentences = set()
    for i in range(len(cosine_matrix)):
        for j in range(i + 1, len(cosine_matrix)):
            if cosine_matrix[i][j] > threshold:
                similar_sentences.add(j)
    
    # 移除相似的句子
    filtered_sentences = [sentence for i, sentence in enumerate(sentences) if i not in similar_sentences]
    
    # 将句子重新组合为文本
    return '. '.join(filtered_sentences) + '.'


def remove_here_is(text):
    pattern = re.compile(r"^(Here is .*?:|.* here is.*?:|Here's .*?:|.* here's.*?:)\s*", re.IGNORECASE)
    return re.sub(pattern, "", text)

def remove_bold_markers(text):
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)


def clean_text(text):
    text = remove_emojis(text)
    text = remove_garbage(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_extra_whitespace(text)
    text = remove_non_ascii(text)
    text = remove_bold_markers(text)
    try:
        text = remove_similar_sentences_tfidf(text)
    except:
        pass
    return text


if __name__ == '__main__':

    with open(os.path.join(work_dir, 'data/Yelp', f'Yelp_rating.pkl'), 'rb') as f:
        trn_set = pickle.load(f)
        tst_set = pickle.load(f)
        dataset = pickle.load(f)
    print(dataset.item_feat)    

    characteristic = json.load(open(f'summarized_review.json'))
    characteristic = pd.DataFrame(
                        list(characteristic.items()),
                        columns=['business_id', 'characteristic']
                    )
    characteristic['business_id'] = characteristic['business_id'].map(
                                    lambda x: dataset.field2token2idx['business_id'].get(x, -1))
    
    dataset.item_feat = pd.merge(
                            dataset.item_feat, 
                            characteristic, 
                            on='business_id', 
                            how='left'
                        ).fillna({'characteristic': ""})
    dataset.item_feat['characteristic'] = dataset.item_feat['characteristic'].map(
                                            lambda x: remove_here_is(x))
    print(dataset.item_feat)

    text_fields = ['name', 'categories', 'characteristic']
    text = dataset.item_feat['name'] + '. ' + \
            dataset.item_feat['categories'] + '. ' + \
            dataset.item_feat['characteristic']
    text = text.map(lambda x: clean_text(x))

    print(dataset.item_feat)

    dataset.item_feat['text'] = text
    dataset.field2type['text'] = 'text'

    dataset.item_feat.drop(columns=text_fields, inplace=True)

    print(dataset.item_feat)
    
    with open(os.path.join(work_dir, 'data/Yelp', f'filtered_Yelp_rating.pkl'), 'wb') as f:
        pickle.dump(trn_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tst_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        print(len(trn_set), len(tst_set), len(dataset.item_feat))
        # 3229584 743660 150347