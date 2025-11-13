'''
预训练脚本
'''

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.nodes import IterableWrapper
# from torchdata.datapipes.iter import IterableWrapper
from itertools import chain
# import deepspeed
from typing import Optional, List

import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
import swanlab

logger = logging.getLogger(__name__)


# 超参数

@dataclass
class ModelArgument:
    """
    关于模型的参数
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "后训练使用，为预训练模型参数地址"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练使用，Config 文件地址"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练 Tokenizer 地址"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型训练使用的数据类型，推荐 bfloat16"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    关于训练的参数
    """
    train_files: Optional[List[str]] = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "设置的文本块长度"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使用线程数."},
    )


def main():
    # print(f'{ModelArgument=}\n{DataTrainingArguments=}')
    parser = HfArgumentParser((ModelArgument, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.warning(f'模型、数据，训练参数:\n{model_args=}\n{data_args=}\n{training_args=}')

    # TrainingArguments(
    # _n_gpu=0,
    # accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_...gacy_prediction_loop=False,
    # use_liger_kernel=False,
    # use_mps_device=False,
    # warmup_ratio=0.0,
    # warmup_steps=0,
    # weight_decay=0.0,
    # )
    swanlab.init(
        project="pretrain",
        experiment_name="from_scrach",

    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 将日志级别设置为 INFO
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 训练整体情况记录
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查ckp
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        # 自带ckp寻找函数
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"{training_args.output_dir} 非空")
        elif last_checkpoint is None and training_args.resume_from_checkpoint is None:
            logger.info(f'从{last_checkpoint} 回复训练')

    # 随机种子
    set_seed(training_args.seed)

    # 初始化模型
    if model_args.config_name is not None:
        # from scrach
        config = AutoConfig.from_pretrained(model_args.config_name)
        logging.warning(f'你正在从零初始化一个模型!')
        logging.info(f'模型参数配置地址:{model_args.config_name}\n模型参数:{config}')
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f'预训练一个新模型 - Total Size={n_params / 2 ** 20:.2f} M params')
    elif model_args.model_name_or_path is not None:
        logger.warning('你正在初始化一个预训练模型')
        logger.info(f'模型参数地址:{model_args.model_name_or_path}')
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f'继承一个新模型 - Total Size={n_params / 2 ** 20:.2f} M params')
    else:
        logger.error('fconfig_name and model_name_path must be not null')
        raise ValueError('fconfig_name and model_name_path must be not null')

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    logging.info("完成 tokenizer 加载")
    logging.info(f'tokenizer配置地址:{model_args.tokenizer_name}')

    # 记载预训练地址
    ds = load_dataset('json', data_files=data_args.train_files)
    logging.info(f'完成训练集加载')
    logging.info(f'训练集地址:{data_args.train_files}')
    logging.info(f'训练集文件总数:{len(ds["train"])}')

    # 文本otkenize
    colunm_names = list(ds['train'].features)
    logging.info(f'训练集特征:{colunm_names=}')
    text_column_name = "text" if "text" in colunm_names else colunm_names[0]

    # tokenize 函数
    def tokenize_func(examples):
        o = tokenizer([i for i in examples[text_column_name]])
        return o

    # 仅主进程进行数据预处理
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = ds.map(
            tokenize_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=colunm_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    # 文本分块
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning("tokenizer 支持大于 1K 的上下文长度，默认设置为 1K")
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(f"设定的块长为 ({data_args.block_size}) ，大于模型的上下文长度"
                           f"将块长设置为模型上下文长度：{tokenizer.model_max_length}.")
            block_size = min(data_args.block_size, tokenizer.model_max_length)


    def group_texts(examples):
        # 将文本段拼接起来
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # 计算拼起来的整体长度
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # 如果长度太长，进行分块
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with training_args.main_process_first(desc="文本分块"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"文本分块到{block_size}",
            batch_size=40000,
        )
        logger.info(f'完成数据预处理')
        train_dataset = lm_datasets['train']

    # 初始化trainer
    trainer = Trainer(model= model,
                      args=training_args,
                      train_dataset=IterableWrapper(train_dataset),
                      tokenizer=tokenizer,
                      data_collator=default_data_collator,)

    # 从ckp加载
    ckp = None
    if training_args.resume_from_checkpoint is not None:
        ckp = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        ckp = last_checkpoint

    logging.info('开始训练!')
    res=  trainer.train(resume_from_checkpoint=ckp)
    trainer.save_model()



if __name__ == '__main__':
    main()
