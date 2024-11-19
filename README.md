# GRE: General Recommendation-Oriented Text Embedding

## Training Dataset

- [AmazonReviews2023](https://amazon-reviews-2023.github.io/)
- [Bili](https://github.com/westlake-repl/NineRec)
- [GoogleLocalData](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/)
- [Steam](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)

### Preprocess for pre-training 

```bash
cd data_pretrain
bash AmazonReviews2023.sh
bash Bili.sh
bash GoogleLocalData.sh
```

### Preprocess for fine-tuning

For each training dataset, execute `process.py`, `filter.py` and `pair.ipynb`

## Test Dataset

- [AmazonReviews2023](https://amazon-reviews-2023.github.io/)
- [Bili](https://github.com/westlake-repl/NineRec)
- [GoodReads](https://www.kaggle.com/datasets/pypiahmad/goodreads-book-reviews1)
- [GoogleLocalData](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/)
- [Yelp](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)

For each test dataset, execute `process.py` and `filter.py`

Notice: for Goodreads, GoogleLocalData, Yelp, change the `low_rating_thres` in config's `*.yaml` to `~` for retrieval

Processed data can be downloaded [here](https://rec.ustc.edu.cn/share/c2ee4a40-5adc-11ef-8048-bf04770908b7).

## Pre-training

```bash
cd RetroMAE/src/pretrain
bash runmix.sh
```

## Fine-tuning

```bash
bash contrastive.py
```

## Benchmark

### Ranking

for naive:

```bash
bash ctr.sh [CUDA_ID_0] [CUDA_ID_1] [DATASET_NAME]
```

for text-embedding-enhanced:

```bash
bash ctrwlm.sh [CUDA_ID_0] [CUDA_ID_1] [DATASET_NAME] [TEXT_EMBEDDING_PATH] [SAVE_PREFIX]
```

### Retrieval

for naive:

```bash
cd RecStudio
bash gre.sh SASRec [DATASET_PKL_PATH] [TEXT_EMBEDDING_PATH]
```

for text-embedding-enhanced:

```bash
cd RecStudio
bash gre.sh TE_ID_SASRec [DATASET_PKL_PATH] [TEXT_EMBEDDING_PATH]
```

### Configuration

You can change the configuration in `din.yaml`, `sasrec.yaml` and `te_id_sasrec.yaml` to search the best results.

