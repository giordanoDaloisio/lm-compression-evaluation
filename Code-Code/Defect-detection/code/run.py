# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import time
import csv
import copy

import numpy as np
import torch
import torch.ao.quantization
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
import torch.nn.utils.prune as prune
import json

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model

cpu_cont = multiprocessing.cpu_count()
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)
from optimum.quanto import qint8, qint4, qfloat8, quantize, freeze, Calibration
from torchinfo import summary
from distilled_dataset import DistilledDataset

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    ),
}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        input_tokens,
        input_ids,
        idx,
        label,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = " ".join(js["func"].split())
    code_tokens = tokenizer.tokenize(code)[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js["idx"], js["target"])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info(example)
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info(
                    "input_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in example.input_tokens]
                    )
                )
                logger.info(
                    "input_ids: {}".format(" ".join(map(str, example.input_ids)))
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(
            self.examples[i].label
        )


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.max_steps * 0.1,
        num_training_steps=args.max_steps,
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    scheduler_last = os.path.join(checkpoint_last, "scheduler.pt")
    optimizer_last = os.path.join(checkpoint_last, "optimizer.pt")
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    # Initialize early stopping parameters at the start of training
    early_stopping_counter = 0
    best_loss = None

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    logging_loss = tr_loss
                    tr_nb = global_step

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):

                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(
                            args, model, tokenizer, eval_when_training=True
                        )
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                        # Save model checkpoint

                    if results["eval_acc"] > best_acc:
                        best_acc = results["eval_acc"]
                        logger.info("  " + "*" * 20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = "checkpoint-best-acc"
                        output_dir = os.path.join(
                            args.output_dir, "{}".format(checkpoint_prefix)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        output_dir = os.path.join(output_dir, "{}".format("model.bin"))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

        # Calculate average loss for the epoch
        avg_loss = train_loss / tr_num

        # Check for early stopping condition
        if args.early_stopping_patience is not None:
            if best_loss is None or avg_loss < best_loss - args.min_loss_delta:
                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping")
                    break  # Exit the loop early


def evaluate(
    args, model, tokenizer, time_log="", time_dir="", eval_when_training=False
):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    if args.vocab_size:
        eval_dataset = DistilledDataset(
            args, args.vocab_size, args.eval_data_file, logger
        )
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            num_workers=8,
            pin_memory=True,
        )

    else:
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        # Note that DistributedSampler samples randomly
        eval_sampler = (
            SequentialSampler(eval_dataset)
            if args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            num_workers=4,
            pin_memory=True,
        )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    inf_times = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            start = time.time()
            lm_loss, logit = model(inputs, label)
            end = time.time()
            inf_time = end - start
            inf_times.append(inf_time)
            if time_dir != "" and time_log != "":
                with open(os.path.join(time_dir, time_log), "a") as f:
                    write = csv.writer(f)
                    write.writerow([inf_time])
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    if "train_unlabel" in args.eval_data_file:
        np.save("../dataset/preds_unlabel_train", logits)
    preds = logits[:, 0] > 0.5
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
    }
    avg_time = np.mean(inf_times)
    logger.info("Average time: " + str(avg_time))
    return result


def test(args, model, tokenizer, time_log="", time_folder=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    if args.vocab_size:
        eval_dataset = DistilledDataset(
            args, args.vocab_size, args.test_data_file, logger
        )
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
            num_workers=8,
            pin_memory=True,
        )
    else:

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
        # Note that DistributedSampler samples randomly
        eval_sampler = (
            SequentialSampler(eval_dataset)
            if args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # GPU WARM-UP
    if args.device == "cuda":
        for i, batch in enumerate(eval_dataloader):
            if i < 5:
                inputs = batch[0].to(args.device)
                label = batch[1].to(args.device)
                with torch.no_grad():
                    logit = model(inputs)
            else:
                break

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    inf_times = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            if args.device == "cuda":
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                starter.record()
                logit = model(inputs)
                ender.record()
                torch.cuda.synchronize()
                inf_time = starter.elapsed_time(ender)
            else:
                starter = time.time()
                logit = model(inputs)
                ender = time.time()
                inf_time = ender - starter
            inf_times.append(inf_time)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            with open(os.path.join(time_folder, time_log), "a") as f:
                csv_f = csv.writer(f)
                csv_f.writerow([inf_time])
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    if args.quantize:
        pred_name = f"predictions_quant_{args.device}.txt"
    elif args.quantize4:
        pred_name = f"predictions_quant4_{args.device}.txt"
    elif args.quantizef8:
        pred_name = f"predictions_quantf8_{args.device}.txt"
    elif args.prune6:
        pred_name = f"predictions_prune6_{args.device}.txt"
    elif args.prune4:
        pred_name = f"predictions_prune4_{args.device}.txt"
    elif args.prune:
        pred_name = f"predictions_prune_{args.device}.txt"
    elif args.prune_local:
        pred_name = f"predictions_prune_local_{args.device}.txt"
    else:
        pred_name = f"predictions_{args.device}.txt"
    with open(os.path.join(args.output_dir, pred_name), "w") as f:
        for example, pred in zip(eval_dataset.examples, preds):
            if pred:
                f.write(example.idx + "\t1\n")
            else:
                f.write(example.idx + "\t0\n")
    logger.info("Average inference time: " + str(np.mean(inf_times)))


def calibrate(args, model, tokenizer):
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    model.eval()
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        with torch.no_grad():
            model(inputs)


def print_model_size(model):
    torch.save(model.state_dict(), "tmp.p")
    logger.info("Size (MB): " + str(os.path.getsize("tmp.p") / 1e6))
    os.remove("tmp.p")


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The model architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument("--model", help="Name of the model file to use.")

    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--epoch", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--min_loss_delta",
        type=float,
        default=0.001,
        help="Minimum change in the loss required to qualify as an improvement.",
    )
    parser.add_argument(
        "--dropout_probability", type=float, default=0, help="dropout probability"
    )

    parser.add_argument("--job_id", type=str, help="Name of the file to log time")
    parser.add_argument("--quantize_dynamic", action="store_true")
    parser.add_argument("--quantize_static", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quantize4", action="store_true")
    parser.add_argument("--quantizef8", action="store_true")
    parser.add_argument("--prune_local", action="store_true")
    parser.add_argument("--prune6", action="store_true")
    parser.add_argument("--prune4", action="store_true")
    parser.add_argument("--prune", action="store_true")

    parser.add_argument("--attention_heads", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--intermediate_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--vocab_size", type=int)

    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    args.device = device
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    if args.quantize_dynamic:
        logfile = f"quantize_dyn_times_{args.job_id}_{args.model_name_or_path.split('/')[0]}.csv"
    elif args.quantize:
        logfile = (
            f"quantize_times_{args.job_id}_{args.model_name_or_path.split('/')[0]}.csv"
        )
    elif args.quantize4:
        logfile = f"quantize4_{args.job_id}_{args.model_name_or_path.split('/')[0]}.csv"
    elif args.prune6:
        logfile = (
            f"prune6_times_{args.job_id}_{args.model_name_or_path.split('/')[0]}.csv"
        )
    elif args.prune4:
        logfile = (
            f"prune4_times_{args.job_id}_{args.model_name_or_path.split('/')[0]}.csv"
        )
    elif args.prune:
        logfile = (
            f"prune2_times_{args.job_id}_{args.model_name_or_path.split('/')[0]}.csv"
        )
    else:
        logfile = f"times_{args.job_id}_{args.model_name_or_path.split('/')[0]}.csv"
    time_dir = os.path.join(args.output_dir, "times")
    os.makedirs(time_dir, exist_ok=True)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, "pytorch_model.bin")
        args.config_name = os.path.join(checkpoint_last, "config.json")
        idx_file = os.path.join(checkpoint_last, "idx_file.txt")
        with open(idx_file, encoding="utf-8") as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, "step_file.txt")
        if os.path.exists(step_file):
            with open(step_file, encoding="utf-8") as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info(
            "reload model from {}, resume from {} epoch".format(
                checkpoint_last, args.start_epoch
            )
        )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if (
        args.attention_heads
        and args.hidden_dim
        and args.intermediate_size
        and args.vocab_size
        and args.n_layers
    ):
        config.num_attention_heads = args.attention_heads
        config.hidden_size = args.hidden_dim
        config.intermediate_size = args.intermediate_size
        config.vocab_size = args.vocab_size
        config.num_hidden_layers = args.n_layers
        config.hidden_dropout_prob = 0.5
        config.num_labels = 2
    else:
        config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.block_size <= 0:
        args.block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path and not args.vocab_size:

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        # For DISTILLATION
        model = model_class(config)

    model = Model(model, config, tokenizer, args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)

    # Evaluation
    if args.do_eval or args.do_test:

        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=device))
        model.to(args.device)

        if args.quantize:
            logger.info("********** Apply Quantization qint8 **********")
            quantize(model, weights=qint8, activations=qint8)
            with Calibration():
                logger.info("*********** Calibrate **************")
                calibrate(args, model, tokenizer)
            freeze(model)

        if args.quantize4:
            logger.info("********** Apply Quantization qint4 **********")
            quantize(model, weights=qint4, activations=qint4)
            with Calibration():
                logger.info("*********** Calibrate **************")
                calibrate(args, model, tokenizer)
            freeze(model)

        if args.quantizef8:
            logger.info("********** Apply Quantization qfloat8 **********")
            quantize(model, weights=qfloat8, activations=qfloat8)
            with Calibration():
                logger.info("*********** Calibrate **************")
                calibrate(args, model, tokenizer)
            freeze(model)

        if args.prune6:
            logger.info("******* Apply Pruning 0.6 ***********")
            parameters_to_prune = []
            for layer in model.encoder.roberta.encoder.layer:
                parameters_to_prune.append((layer.attention.self.query, "weight"))
                parameters_to_prune.append((layer.attention.self.key, "weight"))
                parameters_to_prune.append((layer.attention.self.value, "weight"))
                parameters_to_prune.append((layer.attention.self.query, "bias"))
                parameters_to_prune.append((layer.attention.self.key, "bias"))
                parameters_to_prune.append((layer.attention.self.value, "bias"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.6,
            )
            for module, param in parameters_to_prune:
                prune.remove(module, param)

        if args.prune4:
            logger.info("******* Apply Pruning 0.4 ***********")
            parameters_to_prune = []
            for layer in model.encoder.roberta.encoder.layer:
                parameters_to_prune.append((layer.attention.self.query, "weight"))
                parameters_to_prune.append((layer.attention.self.key, "weight"))
                parameters_to_prune.append((layer.attention.self.value, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.4,
            )
            for module, param in parameters_to_prune:
                logger.info(prune.is_pruned(module))
                prune.remove(module, param)
                logger.info(prune.is_pruned(module))

        if args.prune:
            logger.info("******* Apply Pruning 0.2 ***********")
            parameters_to_prune = []
            for layer in model.encoder.roberta.encoder.layer:
                parameters_to_prune.append((layer.attention.self.query, "weight"))
                parameters_to_prune.append((layer.attention.self.key, "weight"))
                parameters_to_prune.append((layer.attention.self.value, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.2,
            )
            for module, param in parameters_to_prune:
                prune.remove(module, param)

        print_model_size(model)
        summary(model, verbose=2)

        results = {}
        if args.do_eval and args.local_rank in [-1, 0]:

            result = evaluate(args, model, tokenizer, logfile, time_dir)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key], 4)))

        if args.do_test and args.local_rank in [-1, 0]:
            test(args, model, tokenizer, logfile, time_dir)

        return results


if __name__ == "__main__":
    main()
