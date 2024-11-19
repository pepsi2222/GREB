import os
import re
import sys
sys.path.append('.')
sys.path.append('RecStudio')
import pickle
import torch
import copy
import numpy as np
import pandas as pd
from pydantic.utils import deep_update
from utils.utils import get_model
from utils.argument import ModelArguments
import transformers
from transformers import (
    TrainingArguments, 
    EarlyStoppingCallback, 
    AutoTokenizer,
    AutoConfig,
    AutoModel, 
    Trainer,
    set_seed)
from RecStudio.recstudio.utils import parser_yaml, color_dict_normal
from dataset import  Collator, CTRDataset
import logging
from utils.metrics import compute_metrics_for_ctr
from data.MIND.process import MINDSeqDataset
from module.trainer import TrainerWithCustomizedPatience
from peft import PeftModel, PeftConfig

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
                        
def run_ctr_with_text(model : str, dataset : str, mode : str = 'light', **kwargs):
    dataset_specific = parser_yaml('data/dataset_specific.yaml')[dataset]
    dataset_path = os.path.join(os.getenv('DATA_MOUNT_DIR'), dataset_specific['path'])
    dataset_specific['naive_ctr']['train']['output_dir'] = dataset_specific['naive_ctr']['train']['output_dir']#.replace('ctr', f'{kwargs["prefix"]}_ctr')

    trunc = dataset_specific['max_behavior_len']

    # Datasets
    with open(dataset_path, 'rb') as f:
        trn_data = pickle.load(f)
        val_data = pickle.load(f)
        dataset = pickle.load(f)

    # Model arguments
    model_class, model_conf = get_model(model)

    # Arguments
    conf =  deep_update(parser_yaml('config/transformers.yaml'), parser_yaml('config/ctr.yaml'))
    conf = deep_update(conf, model_conf)
    conf = deep_update(conf, dataset_specific['naive_ctr'])
    conf['train']['label_names'] = [dataset.frating]

    if mode == 'debug':
        conf['train']['use_cpu'] = True
        conf['train']['dataloader_num_workers'] = 0
        conf['train']['fp16'] = False

    if kwargs is not None:
        conf['train'].update({k: v for k, v in kwargs.items() if k in conf['train']})
        conf['model'].update({k: v for k, v in kwargs.items() if k in conf['model']})
        conf['ctr_model'].update({k.replace('ctr_model_', ''): v for k, v in kwargs.items() if k.replace('ctr_model_', '') in conf['ctr_model']})
        conf.update({k: v for k, v in kwargs.items() if k in conf})

    conf['train']['learning_rate'] = model_conf['ctr_model']['learning_rate']
    conf['train']['weight_decay'] = model_conf['ctr_model']['weight_decay']
    conf['train']['lr_scheduler_type'] = model_conf['ctr_model']['scheduler']   # linear / cosine / cosine_with_restarts / polynomial / constant / constant_with_warmup / inverse_sqrt / reduce_lr_on_plateau

    # basename = 'no_hidden'
    # basename += f'-trunc{trunc}_factor{conf["factor"]}_lr{conf["train"]["learning_rate"]}_bs{conf["train"]["per_device_train_batch_size"]}_wd{conf["train"]["weight_decay"]}_dropout{conf["ctr_model"]["dropout"]}_lm_dropout{conf["language_model"]["dropout"]}'
    # if "mlp_layer" in conf["ctr_model"]:
    #     basename += f'_mlp{conf["ctr_model"]["mlp_layer"]}'
    # if conf['ctr_model']['batch_norm']:
    #     basename += '_bn'
    # # basename += '_nohidden'
    # # basename += '_lastdropout'
    # if conf['peft_model'] is None and 'contrastive' in conf["model"]["model_name_or_path"]:
    #     parent_dir = re.search('contrastive.*', os.path.dirname(conf["model"]["model_name_or_path"])).group()   
    # elif conf['peft_model'] is not None and 'supervised' in conf["peft_model"]:
    #     parent_dir = re.search('supervised.*', os.path.dirname(conf["peft_model"])).group() 
    # elif conf['peft_model'] is None and 'supervised' in conf["model"]["model_name_or_path"]:
    #     parent_dir = re.search('supervised.*', os.path.dirname(conf["model"]["model_name_or_path"])).group()     
    # elif conf['peft_model'] is None and 'mlm' in conf["model"]["model_name_or_path"]:
    #     parent_dir = re.search('mlm.*', os.path.dirname(conf["model"]["model_name_or_path"])).group() 
    # else:
    #     parent_dir = os.path.basename(conf["model"]["model_name_or_path"])
    #     if parent_dir == 'language_model':
    #         parent_dir = os.path.basename(os.path.dirname(conf["model"]["model_name_or_path"]))

    conf['train']['output_dir'] = os.path.join(
                                    'test',
                                    # 'final_results',
                                    conf['train']['output_dir']+'_with_text', 
                                    # model, 
                                    # parent_dir,
                                    # basename
                                )
    conf['train']['logging_dir'] = os.path.join(conf['train']['output_dir'], 'tb')

    training_args = TrainingArguments(**conf['train'])
    training_args.patience = conf['patience']
    training_args.factor = conf['factor']

    model_args = ModelArguments(**conf['model'])

    set_seed(training_args.seed)


    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout), 
        ],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    transformers.utils.logging.add_handler(
        logging.FileHandler(os.path.join(training_args.output_dir, f'log.log')))
    transformers.utils.logging.enable_explicit_format()
    logger = transformers.utils.logging.get_logger()
    logger.setLevel(log_level)

    logger.info('****All Configurations****')
    logger.info(color_dict_normal(conf))


    # Config
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)


    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        tokenizer = None
    

    # Language model
    if model_args.model_name_or_path:
        language_model = AutoModel.from_pretrained(
                            model_args.model_name_or_path,
                            from_tf=bool(".ckpt" in model_args.model_name_or_path),
                            config=config,
                            cache_dir=model_args.cache_dir,
                            revision=model_args.model_revision,
                            token=model_args.token,
                            trust_remote_code=model_args.trust_remote_code,
                            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                        )
        if conf['peft_model'] is not None:
            language_model = PeftModel.from_pretrained(language_model, conf['peft_model'])
    else:
        raise ValueError()
    
    

    # Datasets
    dataset.dataframe2tensors()

    dataset_specific = parser_yaml('data/dataset_specific.yaml')
    # prefix = dataset_specific[dataset.name]['prefix']


    trn_dataset = CTRDataset(dataset, trn_data, tokenizer=tokenizer, trunc=trunc)#, prefix=prefix)
    trn_dataset.get_tokenized_text()
    val_dataset = CTRDataset(dataset, val_data, tokenizer=tokenizer, trunc=trunc)
        

    ctr_fields = {dataset.fuid, dataset.fiid, dataset.frating}.union(
                dataset.item_feat.fields).union(dataset.user_feat.fields) - {'text'}
        
    ctr_fields = sorted(list(ctr_fields))

    
    # Callback
    callbacks = [EarlyStoppingCallback(**conf['early_stop'])]

    # Model
    model = model_class(
                config=conf, 
                dataset=dataset, 
                ctr_fields=ctr_fields, 
                language_model=language_model,
                item_text_feat=dataset.item_feat.data['text'],
                tokenizer=tokenizer,
            )   
        
    if model_args.model_name_or_path:
        model.freeze('language_model')
        model.build_text_embedding()#saved_dir=os.path.dirname(training_args.output_dir))
        dataset.item_feat.del_fields(keep_fields=set(dataset.item_feat.fields) - {'text'})
        model.language_model = None
        del language_model

    if conf['pretrained_dir'] is not None:
        logger.info(f"Loaded from {conf['pretrained_dir']}.")
        model.from_pretrained(conf['pretrained_dir'])

    # Trainer
    trainer = TrainerWithCustomizedPatience(
        model=model,
        args=training_args,
        data_collator=Collator(tokenizer),
        train_dataset=trn_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_ctr,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        trn_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trn_metrics = trn_result.metrics
        trn_metrics["train_samples"] = len(trn_dataset)
        trn_metrics["validate_samples"] = len(val_dataset)

        trainer.log_metrics("train", trn_metrics)
        trainer.save_metrics("train", trn_metrics)
        trainer.save_state()

    # Eval
    if training_args.do_eval:
        val_metrics = trainer.evaluate(val_dataset)
        val_metrics["validate_samples"] = len(val_dataset)
        trainer.log_metrics("eval", val_metrics)
        trainer.save_metrics("eval", val_metrics)


if __name__ == '__main__':

    run_ctr_with_text(
            model='DIN',
            dataset=sys.argv[1],
            model_name_or_path=sys.argv[2],
            # tokenizer_name='/data1/home/xingmei/GRE/output/e5-RetroMAE_pretrained/checkpoint-9765',
            # prefix=sys.argv[3]
            # tokenizer_name='/data1/home/xingmei/GRE/supervised-hardprompt/perdevicebs128-lora-temperature0.1-lr3e-05-gradacc10-expand_token',#'/data1/home/xingmei/GRE/output/e5-RetroMAE_pretrained/checkpoint-9765',
            # peft_model=sys.argv[1] if sys.argv[1] != 'None' else None,
            
            # resume_from_checkpoint=False,
            # pretrained_dir=sys.argv[3],
            # ctr_model_learning_rate=float(sys.argv[4])
        )