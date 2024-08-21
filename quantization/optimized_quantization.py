# multihead attention
import copy, torch, math, numpy as np, pandas as pd, logging, sys
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from transformers import AutoModelForCausalLM
from base_modules import *

def initialize_model(hyperparameters):
    ## initializing toy model 
    hidden_states = 768
    seq_len = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # embedding
    tokenizer_checkpoint = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    tokenizer.pad_token = "[PAD]"

    model_for_embedding = AutoModelForCausalLM.from_pretrained(tokenizer_checkpoint)
    model_for_embedding.resize_token_embeddings(len(tokenizer))

    token_embedding = model_for_embedding.transformer.wte
    positional_embedding = model_for_embedding.transformer.wpe
    embedding = Embedding(token_embedding, positional_embedding)

    for params in embedding.parameters():
        params.requires_grad = False
        print(params.requires_grad)

    # tokenEmbedding = TokenEmbedding(hidden_states, len(tokenizer))
    # positionalEmbedding = PositionalEmbedding(hidden_states, seq_len, device)
    # embedding = Embedding(tokenEmbedding, copy.deepcopy(positionalEmbedding))

    # attention layer
    attention_dim = 64
    num_head = 4
    qkv_fc = nn.Linear(in_features=hidden_states, out_features=attention_dim, bias=True)
    out_fc = nn.Linear(in_features=attention_dim, out_features=hidden_states, bias=True)
    attention = MultiheadAttention(attention_dim, num_head, qkv_fc, out_fc)

    # FFNN
    d_ffnn = 1024
    norm_eps = 1e-05
    ffnn = PositionwiseFFNN(hidden_states, d_ffnn)
    layerNorm = nn.LayerNorm(hidden_states, eps = norm_eps)

    # decoder(AutoRegressive model)
    dropout_rate = 0.1
    num_layers = 1
    decoderBlock = EncoderBlock(attention, ffnn, layerNorm, dropout_rate)
    decoder = Encoder(decoderBlock, num_layers)

    # generator
    generator = Generator(len(tokenizer), hidden_states) 

    #transformer
    model = Transformer(embedding, decoder, generator, tokenizer)

    return model, tokenizer


def set_sft_config():

        sft_config_dict = {
            "output_dir": "/home/hyohyeongjang/quantization/outputs",
            # "num_train_epochs": 2,
            "per_device_train_batch_size": 128,
            "per_device_eval_batch_size": 128,
            "auto_find_batch_size": True,
            "gradient_accumulation_steps": 10,
            "learning_rate": 3e-5,
            "lr_scheduler_type": "linear", # default
            "weight_decay": 0.001,
            "eval_strategy": "steps",
            "eval_steps": 100,
            "eval_accumulation_steps": None, # default
            "save_strategy": "steps", # default
            "save_steps": 100,
            "logging_strategy": "steps",
            "save_total_limit": 1,
            "load_best_model_at_end": False,
            "metric_for_best_model": "eval_loss",
            "dataloader_num_workers": 10,
            "seed": 42,
            "use_cpu" : True
            
        }

        sft_config = SFTConfig(
            **sft_config_dict,
            bf16=True,
            remove_unused_columns=True,
            group_by_length=False,
            disable_tqdm=False,
        )

        return sft_config




def formatting_func(examples):
    return [examples['output'][i] for i in range(len(examples))]

if __name__ == "__main__":

    logger = logging.getLogger("main")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(f"[%(asctime)s][%(levelname)s] %(message)s")

    streamer = logging.StreamHandler(sys.stdout)
    streamer.setFormatter(formatter)
    streamer.setLevel(logging.INFO)
    logger.addHandler(streamer)

    
    filer = logging.FileHandler(
        filename=f"runs/train_log_{datetime.now()}.log", mode="w", encoding="utf-8"
    )
    filer.setFormatter(formatter)
    filer.setLevel(logging.DEBUG)
    logger.addHandler(filer)

    logger.info("Logger prepared")
    logger.info(f"Logs will be documented to: {filer.baseFilename}")

    
    model, tokenizer = initialize_model(None)
    num_params = sum(p.numel() for p in model.parameters())
    logger.debug(f"num_params: {format(num_params, ',d')}")
    
    ## initialize model

    #others
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # dataset
    dataset = load_dataset("csv", sep=",", data_files="/node_storage2/data_llm_kr/data_it_eval_240724.csv", keep_default_na = False)
    dataset = dataset['train'].train_test_split(test_size=0.2)
    train_dataset = dataset['train']#['output']
    # train_dataset = train_dataset.remove_columns(['task','input','instruction','option'])
    # train_dataset = tokenizer(train_dataset, padding=True, max_length=seq_len, truncation=True, return_tensors="pt")

    valTestDataset = dataset['test'].train_test_split(test_size = 0.5)
    eval_dataset = valTestDataset['train']#['output']
    # eval_dataset = eval_dataset.remove_columns(['task','input','instruction','option'])
    # valid_dataset = tokenizer(valid_dataset, padding=True, max_length=seq_len, truncation=True, return_tensors="pt")

    test_dataset = valTestDataset['test']#['output']
    # test_dataset = test_dataset.remove_columns(['task','input','instruction','option'])
    # test_dataset = tokenizer(test_dataset, padding=True, max_length=seq_len, truncation=True, return_tensors="pt")
    print("train, valid, testset size:", len(train_dataset), len(eval_dataset), len(test_dataset))

    logger.debug("datasets are from: /node_storage2/data_llm_kr/data_it_eval_240724.csv")

    logger.debug(f"{test_dataset}")


    sft_config = set_sft_config()
    trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            formatting_func=formatting_func,
            
            data_collator=collator, # lets try with default collator
            max_seq_length=512,
            dataset_num_proc=10,
        )


    logger.debug("start")
    
    trainer.train()

    logger.debug("done")

    

