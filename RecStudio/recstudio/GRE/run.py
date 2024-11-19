import os, datetime, torch
from typing import *
from recstudio.utils import *
from recstudio import LOG_DIR
import sys
from datetime import datetime
import pickle


DATASET_CONFIG_DICT = {
    'AmazonReviews2023/dataset/filtered_Arts_Crafts_and_Sewing.pkl': {
        'model': {'layer_num': 3, 'embed_dim': 256},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Automotive.pkl': {
        'model': {'layer_num': 3, 'embed_dim': 128},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Baby_Products.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 256},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Cell_Phones_and_Accessories.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Industrial_and_Scientific.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 128},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Musical_Instruments.pkl': {
        'model': {'layer_num': 3, 'embed_dim': 256},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Office_Products.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Patio_Lawn_and_Garden.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 256},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Sports_and_Outdoors.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Tools_and_Home_Improvement.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'OnlineRetail/filtered_OnlineRetail.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 128},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Beauty_and_Personal_Care.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'Goodreads/filtered_Goodreads_rating.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Books.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Toys_and_Games.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'AmazonReviews2023/dataset/filtered_Video_Games.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'Bili/dataset/filtered_Bili_Cartoon.pkl': {
        'model': {'layer_num': 3, 'embed_dim': 64},
        'train': {'batch_size': 512}
    },
    'Bili/dataset/filtered_Bili_Dance.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 512}
    },
    'Bili/dataset/filtered_Bili_Food.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 512}
    },
    'Bili/dataset/filtered_Bili_Movie.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 512}
    },
    'Bili/dataset/filtered_Bili_Music.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 128},
        'train': {'batch_size': 512}
    },
    'Bili/dataset/filtered_DY.pkl': {
        'model': {'layer_num': 3, 'embed_dim': 128},
        'train': {'batch_size': 512}
    },
    'Bili/dataset/filtered_KU.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 128},
        'train': {'batch_size': 512}
    },
    'Bili/dataset/filtered_QB.pkl': {
        'model': {'layer_num': 3, 'embed_dim': 64},
        'train': {'batch_size': 512}
    },
    'Bili/dataset/filtered_TN.pkl': {
        'model': {'layer_num': 3, 'embed_dim': 128},
        'train': {'batch_size': 512}
    },
    'Yelp/filtered_Yelp_rating.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    },
    'GoogleLocalData/Food/rating_dataset/filtered_Maine.pkl': {
        'model': {'layer_num': 2, 'embed_dim': 64},
        'train': {'batch_size': 2048}
    }
}



def run(model: str, dataset_path: str, model_config: Dict=None, data_config: Dict=None, 
        model_config_path: str=None, data_config_path: str=None, 
        verbose=True, run_mode='light', **kwargs):

    # load config and update config with nni 
    model_class, model_conf = get_model(model)
    if model_config_path is not None:
        if isinstance(model_config_path, str):
            model_conf = deep_update(model_conf, parser_yaml(model_config_path))
        else:
            raise TypeError(f"expecting `model_config_path` to be str, while get {type(model_config_path)} instead.")

    if model_config is not None:
        if isinstance(model_config, Dict):
            model_conf = deep_update(model_conf, model_config)
        else:
            raise TypeError(f"expecting `model_config` to be Dict, while get {type(model_config)} instead.")

    if kwargs is not None:
        model_conf = deep_update(model_conf, kwargs)

    if run_mode == 'tune':
        model_conf = update_config_with_nni(model_conf)

    # load dataset 
    dataset_path = model_conf['main']['dataset_path']
    with open(os.path.join(os.getenv('DATA_MOUNT_DIR'), dataset_path), 'rb') as f:
        _ = pickle.load(f)
        _ = pickle.load(f)
        dataset = pickle.load(f)
    
        
    # log_path = f"{model}/{dataset.name}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log"
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f")
    log_path = f"{dataset.name}_{timestamp}.log"
    if 'text_encoder' in model_conf:
        if 'all_fields/e5_pretrain' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"e5_pt/{model}/{dataset.name}_{timestamp}.log"
            save_dir = './saved_e5_pt'
        elif 'all_fields3/e5_pretrain' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"saved_e5_no_balance_pt/{model}/{dataset.name}_{timestamp}.log"
            save_dir = './saved_e5_no_balance_pt'
        elif 'contrastive_large' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"contrastive_large/{dataset.name}_{timestamp}.log"
            save_dir = './saved_contrastive_large'
        elif 'contrastive' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"e5_pt_cl/{model}/{dataset.name}_{timestamp}.log"
            save_dir = './saved_e5_pt_cl'
        elif 'e5-base' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"e5-base/{model}/{dataset.name}_{timestamp}.log"
            save_dir = './saved_e5_base'
        elif 'unsup-simcse' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"unsup_simcse/{dataset.name}_{timestamp}.log"
            save_dir = './saved_unsup_simcse'
        elif 'sup-simcse' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"sup_simcse/{dataset.name}_{timestamp}.log"
            save_dir = './saved_sup_simcse'
        elif 'bert-base-uncased' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"bert/{dataset.name}_{timestamp}.log"
            save_dir = './saved_bert'
        elif 'gtr' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"gtr/{dataset.name}_{timestamp}.log"
            save_dir = './saved_gtr'
        elif 'contriever' in model_conf['text_encoder']['model_name_or_path']:
            log_path = f"contriever/{dataset.name}_{timestamp}.log"
            save_dir = './saved_contriever'
    else:
        log_path = f"{model}/{dataset.name}_{timestamp}.log"
        save_dir = './saved'
    model_conf['eval']['save_path'] = save_dir
    logger = get_logger(log_path)
    torch.set_num_threads(model_conf['train']['num_threads'])

    if model_conf['main']['use_dataset_config'] == 1 and (model_conf['main']['dataset_path'] in DATASET_CONFIG_DICT):
        logger.info(f"use dataset config: {DATASET_CONFIG_DICT[model_conf['main']['dataset_path']]}")
        model_conf = deep_update(model_conf, DATASET_CONFIG_DICT[model_conf['main']['dataset_path']])
        
    if not verbose:
        import logging
        logger.setLevel(logging.ERROR)

    logger.info("Log saved in {}.".format(os.path.abspath(os.path.join(LOG_DIR, log_path))))
    
    model = model_class(model_conf)
    # dataset_class = model_class._get_dataset_class()

    # data_conf = {}
    # if data_config_path is not None:
    #     if isinstance(data_config_path, str):
    #         # load dataset config from file
    #         conf = parser_yaml(data_config_path)
    #         data_conf.update(conf)
    #     else:
    #         raise TypeError(f"expecting `data_config_path` to be str, while get {type(data_config_path)} instead.")

    # if data_config is not None:
    #     if isinstance(data_config, dict):
    #         # update config with given dict
    #         data_conf.update(data_config)
    #     else:
    #         raise TypeError(f"expecting `data_config` to be Dict, while get {type(data_config)} instead.")

    # data_conf.update(model_conf['data'])    # update model-specified config
    
    # dataset.config.update(data_conf)
    logger.info(f'dataset_path updated : {dataset_path}')
    datasets = dataset.build(**model_conf['data'])
    logger.info(f"{datasets[0]}")
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
    val_result = model.fit(*datasets[:2], run_mode=run_mode)
    test_result = model.evaluate(datasets[-1])
    return (model, datasets), (val_result, test_result)
