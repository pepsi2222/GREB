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
import operator
from functools import reduce
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
csv.field_size_limit(sys.maxsize)

from transformers import set_seed
set_seed(42)
import time


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

def remove_original_text(text):
    """
    (Translated by Google) This dining environment is very comfortable. The view is great. Even all the workers are very good. Meals. The diet is quite delicious. There are also concerts on holidays. Enjoy the view. Enjoy local flavor music. How enjoyable. The aftertaste is endless.

(Original)
這家用餐環境非常舒適。視野非常棒。連所有工作者服務都非常棒。餐點。飲食相當美味之外。假日還有演奏會。欣賞美景之外。享受當地風味音樂。是多麼享受。回味無窮。
    """
    # 匹配并移除(Translated by Google)前缀
    text = re.sub(r'\(Translated by Google\)\s*', '', text)
    # 匹配并移除(Original)到下一段之间的文本
    pattern = re.compile(r'\(Original\).*', re.DOTALL)
    cleaned_text = pattern.sub('', text)
    return cleaned_text.strip()

def remove_similar_sentences_tfidf(text, threshold=0.7):
    is_list = isinstance(text, list)
    if not is_list:
        # 将文本拆分为句子
        sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    else:
        sentences = text
    
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
    
    if not is_list:
        # 将句子重新组合为文本
        return '. '.join(filtered_sentences) + '.'
    else:
        return filtered_sentences

def clean_text(text):
    text = remove_emojis(text)
    text = remove_garbage(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_extra_whitespace(text)
    text = remove_non_ascii(text)
    text = remove_original_text(text)
    try:
        text = remove_similar_sentences_tfidf(text)
    except:
        pass
    return text



if __name__ == '__main__':

    review = defaultdict(set)
    with open(os.path.join(work_dir, 'data/Yelp/review.json'), 'r') as rf:
        while True:
            datum = rf.readline()
            if not datum:
                break

            try:
                datum = eval(datum)
                if datum['text'] is not None and is_english(datum['text']):
                    review[datum['business_id']].add(clean_text(datum['text']))
            except Exception as e:
                print(e, datum)
                continue

    for k in review:
        review[k] = remove_similar_sentences_tfidf(list(review[k]))

    with open(os.path.join(work_dir, 'data/Yelp/filtered_reviews.json'), 'w') as f:
        json.dump(review, f)
    print('Success!')
