import os
import sys
sys.path.append('.')
sys.path.append('RecStudio')
import pickle
import torch
import copy
import numpy as np
from pydantic.utils import deep_update
from utils.utils import get_model
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
from RecStudio.recstudio.utils import parser_yaml, color_dict_normal
from dataset import CTRDataset, Collator
from utils.metrics import compute_metrics_for_ctr
import logging
import transformers
from data.MIND.process import MINDSeqDataset
from module.trainer import TrainerWithCustomizedPatience


def run_ctr(model : str, dataset: str, mode : str = 'light', **kwargs):
    dataset_specific = parser_yaml('data/dataset_specific.yaml')[dataset]
    dataset_path = os.path.join(os.getenv('DATA_MOUNT_DIR'), dataset_specific['path'])
    # if 'Amazon' in dataset_path:
    #     dataset_path = dataset_path.replace('dataset', '0core_dataset')
    #     dataset_specific['naive_ctr']['train']['output_dir'] = dataset_specific['naive_ctr']['train']['output_dir'].replace('ctr', '0core_ctr')
    # if 'Google' in dataset_path:
    #     if dataset_specific['naive_ctr']['train']['per_device_train_batch_size'] > 4096:
    #         return
    #     else:
    #         dataset_path = dataset_path.replace('dataset', 'rating_dataset')
    #         dataset_specific['naive_ctr']['train']['output_dir'] = dataset_specific['naive_ctr']['train']['output_dir'].replace('ctr', 'rating_ctr')
    
    trunc = dataset_specific['max_behavior_len']

    # Datasets
    with open(dataset_path, 'rb') as f:
        trn_data = pickle.load(f)
        val_data = pickle.load(f)
        dataset = pickle.load(f)


    drop_fields = set(f for f, t in dataset.field2type.items() if t == 'text')
    dataset.dataframe2tensors()
    fields = {dataset.frating}
    if dataset.item_feat is not None:
        dataset.item_feat.del_fields(keep_fields=set(dataset.item_feat.fields) - drop_fields)
        fields = fields.union(dataset.item_feat.fields)

    if dataset.user_feat is not None:
        dataset.user_feat.del_fields(keep_fields=set(dataset.user_feat.fields) - drop_fields)
        fields = fields.union(dataset.user_feat.fields)


    fields = sorted(list(fields))

    trn_dataset = CTRDataset(dataset, trn_data, trunc=trunc)
    val_dataset = CTRDataset(dataset, val_data, trunc=trunc)
    

    # Model arguments
    model_class, model_conf = get_model(model)

    # Training arguments
    conf = deep_update(parser_yaml('config/transformers.yaml'), parser_yaml('config/ctr.yaml'))
    conf = deep_update(conf, model_conf)
    conf = deep_update(conf, dataset_specific['naive_ctr'])
    conf['train']['label_names'] = [dataset.frating]
    if mode == 'debug':
        conf['train']['use_cpu'] = True
        conf['train']['dataloader_num_workers'] = 0
        conf['train']['fp16'] = False

    if kwargs is not None:
        conf['train'].update({k: v for k, v in kwargs.items() if k in conf['train']})
        conf['ctr_model'].update({k.replace('ctr_model_', ''): v for k, v in kwargs.items() if k.replace('ctr_model_', '') in conf['ctr_model']})
        conf.update({k: v for k, v in kwargs.items() if k in conf})

    conf['train']['learning_rate'] = model_conf['ctr_model']['learning_rate']
    conf['train']['weight_decay'] = model_conf['ctr_model']['weight_decay']
    conf['train']['lr_scheduler_type'] = model_conf['ctr_model']['scheduler']   # linear / cosine / cosine_with_restarts / polynomial / constant / constant_with_warmup / inverse_sqrt / reduce_lr_on_plateau

    
    basename = f'trunc{trunc}_factor{conf["factor"]}_lr{conf["train"]["learning_rate"]}_bs{conf["train"]["per_device_train_batch_size"]}_wd{conf["train"]["weight_decay"]}_dropout{conf["ctr_model"]["dropout"]}'
    if "mlp_layer" in conf["ctr_model"]:
        basename += f'_mlp{conf["ctr_model"]["mlp_layer"]}'
    if conf['ctr_model']['batch_norm']:
        basename += '_bn'
    # conf['train']['output_dir'] = os.path.join(conf['train']['output_dir'], model, basename)
    conf['train']['output_dir'] = os.path.join(
                                    f'final_results_{model}',
                                    conf['train']['output_dir'], 
                                    # model, 
                                    # parent_dir,
                                    # basename
                                )
    conf['train']['logging_dir'] = os.path.join(conf['train']['output_dir'], 'tb')

    training_args = TrainingArguments(**conf['train'])
    training_args.patience = conf['patience']
    training_args.factor = conf['factor']

    set_seed(training_args.seed)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    transformers.utils.logging.add_handler(
        logging.FileHandler(
            os.path.join(training_args.output_dir, f'log.log')))
    transformers.utils.logging.enable_explicit_format()
    logger = transformers.utils.logging.get_logger()
    logger.setLevel(log_level)
    logger.info('****All Configurations****')
    logger.info(color_dict_normal(conf))


    # Model
    model = model_class(conf, dataset, fields)
    if conf['pretrained_dir'] is not None:
        logger.info(f"Loaded from {conf['pretrained_dir']}.")
        model.from_pretrained(conf['pretrained_dir'])

    # Trainer
    trainer = TrainerWithCustomizedPatience(
        model=model,
        args=training_args,
        data_collator=Collator(),
        train_dataset=trn_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_ctr,
        callbacks=[EarlyStoppingCallback(**conf['early_stop'])],
    )

    # Training
    if training_args.do_train:
        # assert torch.cuda.device_count() == 2, 'Num of GPUs should be 2 when training naive CTR model.'

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

    run_ctr(
        model='DIN', 
        dataset=sys.argv[1],
        mode='light',
        # pretrained_dir=sys.argv[2],
        # ctr_model_learning_rate=float(sys.argv[4])
        # ctr_model_learning_rate=float(sys.argv[2]),
        # ctr_model_weight_decay=float(sys.argv[3]),
        # per_device_train_batch_size=int(sys.argv[4])
    )