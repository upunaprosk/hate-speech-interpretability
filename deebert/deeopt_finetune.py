# coding=utf-8
"""
Finetuning OPT (decoder-only) with highway early-exit heads for sequence classification.
API and flags mirror the previous BERT highway runner.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc

from transformers import (
    WEIGHTS_NAME,
    OPTConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# === your decoder-only highway model ===
# Make sure highway_opt.py defines OPTForSequenceClassificationHighway (as we discussed).
from highway_opt import OPTForSequenceClassificationHighway


logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------
# Metrics
#-----------------------------------------------------------------------
def compute_metrics(task_name, preds, labels):
    acc = accuracy_score(labels, preds)
    f1_bin = f1_score(labels, preds, average="binary")
    f1_mac = f1_score(labels, preds, average="macro")
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    return {
        "accuracy": acc,
        "f1_binary": f1_bin,
        "f1_macro": f1_mac,
        "pr_auc": pr_auc,
    }

def get_wanted_result(result, key="f1_macro"):
    return result[key]


#-----------------------------------------------------------------------
# Repro
#-----------------------------------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#-----------------------------------------------------------------------
# Train
#-----------------------------------------------------------------------
def train(args, train_dataset, model, tokenizer, train_highway=False):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Optimizer / schedule
    no_decay = ["bias", "LayerNorm.weight"]
    if train_highway:
        # only highway params
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if ("encoder.highway" in n) and not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if ("encoder.highway" in n) and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    else:
        # base model (no highway) + final classifier
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if ("encoder.highway" not in n) and (not any(nd in n for nd in no_decay))],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if ("encoder.highway" not in n) and (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Install apex from https://github.com/NVIDIA/apex to use fp16.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train loop
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    world = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    logger.info("  Total train batch size (parallel, dist & accum) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * world)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)  # (ids, attn, labels)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "train_highway": train_highway,
            }
            outputs = model(**inputs)
            loss = outputs["loss"]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(f"eval_{key}", value, global_step)
                    # use get_last_lr in recent HF/torch
                    try:
                        lr = scheduler.get_last_lr()[0]
                    except Exception:
                        lr = scheduler.get_lr()[0]
                    tb_writer.add_scalar("lr", lr, global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / max(1, global_step)


#-----------------------------------------------------------------------
# Eval
#-----------------------------------------------------------------------
def evaluate(args, model, tokenizer, prefix="", output_layer=-1, eval_highway=False):
    results = {}

    eval_task = args.task_name
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    # exit-layer stats
    num_layers = getattr(model, "num_layers", model.config.num_hidden_layers)
    exit_layer_counter = {i + 1: 0 for i in range(num_layers)}

    st = time.time()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }
            if output_layer >= 0:
                inputs["output_layer"] = output_layer

            outputs = model(**inputs)
            if eval_highway:
                exit_layer_counter[outputs["exit_layer"]] += 1

            tmp_eval_loss = outputs["loss"]
            logits = outputs["logits"]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_time = time.time() - st
    print("Eval time:", eval_time)

    eval_loss = eval_loss / max(1, nb_eval_steps)
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(eval_task, preds, out_label_ids)
    print(result)
    results.update(result)

    if eval_highway:
        print("Exit layer counter", exit_layer_counter)
        actual_cost = sum([l * c for l, c in exit_layer_counter.items()])
        full_cost = len(eval_dataloader) * num_layers
        print("Expected saving", actual_cost / max(1, full_cost))
        avg_exit_layer = sum(l * c for l, c in exit_layer_counter.items()) / max(1, sum(exit_layer_counter.values()))
        print(f"Average Exit Layer: {avg_exit_layer:.2f}")
        result["avg_exit_layer"] = avg_exit_layer

        if args.early_exit_entropy >= 0:
            save_fname = os.path.join(
                args.plot_data_dir,
                args.model_name_or_path.replace("/", "_"),
                f"entropy_{args.early_exit_entropy}.npy",
            )
            os.makedirs(os.path.dirname(save_fname), exist_ok=True)
            print_result = get_wanted_result(result)
            np.save(save_fname, np.array([exit_layer_counter, eval_time, actual_cost / max(1, full_cost), print_result]))

    # write eval file
    prefix_dir = os.path.join(eval_output_dir, prefix) if prefix else eval_output_dir
    os.makedirs(prefix_dir, exist_ok=True)
    output_eval_file = os.path.join(prefix_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


#-----------------------------------------------------------------------
# Data
#-----------------------------------------------------------------------
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    # Assumes CSVs with columns: sentence, label (0/1).
    split = "test" if evaluate else "train"
    path = os.path.join(args.data_dir, f"{task}_{split}.csv")

    import pandas as pd
    df = pd.read_csv(path)
    texts = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length,
        return_tensors="pt",
    )
    all_input_ids = enc["input_ids"].long()
    all_attention_mask = enc["attention_mask"].long()
    all_labels = torch.tensor(labels, dtype=torch.long)

    # For OPT we do NOT need token_type_ids; dataset is (ids, mask, labels)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    return dataset


#-----------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--data_dir", type=str, required=True, help="Input data dir with <task>_{train,test}.csv")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="e.g., facebook/opt-125m or a checkpoint dir")
    parser.add_argument("--task_name", type=str, required=True, help="Task name used to locate CSVs")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write checkpoints/predictions")
    parser.add_argument("--plot_data_dir", default="./plotting/", type=str, help="Where to save plotting artifacts")

    # Other
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")  # ignored for OPT but kept for CLI parity
    parser.add_argument("--eval_each_highway", action="store_true")
    parser.add_argument("--eval_after_first_stage", action="store_true")
    parser.add_argument("--eval_highway", action="store_true")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--early_exit_entropy", default=-1.0, type=float, help="Entropy threshold for early exit")

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_all_checkpoints", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--server_ip", type=str, default="")
    parser.add_argument("--server_port", type=str, default="")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) exists and is not empty. Use --overwrite_output_dir to override."
        )

    # DDP debugger (optional)
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # CUDA & DDP
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed: %s, fp16: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
    )

    # Seed
    set_seed(args)

    # Labels
    args.task_name = args.task_name.lower()
    args.output_mode = "classification"
    label_list = [0, 1]
    num_labels = len(label_list)

    # Load config, tokenizer, model
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    config = OPTConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = OPTForSequenceClassificationHighway.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # expose num_layers attribute for downstream code
    model.num_layers = config.num_hidden_layers

    # set early-exit thresholds
    model.set_early_exit_entropy(args.early_exit_entropy)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Train
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, train_highway=False)
        logger.info("First stage (base) — global_step = %s, avg loss = %s", global_step, tr_loss)

        if args.eval_after_first_stage:
            result = evaluate(args, model, tokenizer, prefix="")
            _ = get_wanted_result(result)

        # train highway heads
        _, _ = train(args, train_dataset, model, tokenizer, train_highway=True)

        # save
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # reload
        model = OPTForSequenceClassificationHighway.from_pretrained(args.output_dir)
        model.num_layers = model.config.num_hidden_layers
        model.set_early_exit_entropy(args.early_exit_entropy)
        model.to(args.device)

    # Evaluate
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = os.path.basename(checkpoint) if "checkpoint" in checkpoint else ""

            model = OPTForSequenceClassificationHighway.from_pretrained(checkpoint)
            model.num_layers = model.config.num_hidden_layers
            model.set_early_exit_entropy(args.early_exit_entropy)
            model.to(args.device)

            result = evaluate(args, model, tokenizer, prefix=prefix, eval_highway=args.eval_highway)
            print_result = get_wanted_result(result)
            print("Result:", print_result)
            acc = result["accuracy"] * 100
            f1m = result["f1_macro"] * 100
            print(f"Acc: {acc:.2f} | F1-macro: {f1m:.2f}")

            if args.eval_each_highway:
                # evaluate each intermediate layer by forcing output_layer
                last_layer_results = print_result
                each_layer_results = []
                for i in range(model.num_layers):
                    logger.info("\n")
                    _result = evaluate(args, model, tokenizer, prefix=prefix, output_layer=i, eval_highway=args.eval_highway)
                    if i + 1 < model.num_layers:
                        each_layer_results.append(get_wanted_result(_result))
                each_layer_results.append(last_layer_results)
                save_fname = os.path.join(args.plot_data_dir, args.model_name_or_path.replace("/", "_"), "each_layer.npy")
                os.makedirs(os.path.dirname(save_fname), exist_ok=True)
                np.save(save_fname, np.array(each_layer_results))

            result = dict((k + "_{}".format(prefix), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
