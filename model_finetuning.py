import os, json, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import evaluate
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, default_data_collator, EvalPrediction
from huggingface_hub import HfApi, login


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_table_auto(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")


def load_split(path, text_col, label_col):
    df = read_table_auto(path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


class BinaryHateDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.enc["input_ids"][i]), "attention_mask": torch.tensor(self.enc["attention_mask"][i]), "labels": self.labels[i]}


ACC = evaluate.load("accuracy", keep_in_memory=True)
F1  = evaluate.load("f1", keep_in_memory=True)

def compute_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    logits = np.asarray(logits)
    labels = np.asarray(p.label_ids)
    preds  = logits.argmax(axis=-1)
    acc = ACC.compute(predictions=preds, references=labels)["accuracy"]
    f1_bin = F1.compute(predictions=preds, references=labels, average="binary")["f1"]
    f1_mac = F1.compute(predictions=preds, references=labels, average="macro")["f1"]
    if logits.shape[1] == 2:
        pos_scores = logits[:, 1]
        precision, recall, _ = precision_recall_curve(labels, pos_scores)
        pr_auc = auc(recall, precision)
    else:
        pr_auc = float("nan")
    return {"accuracy": acc, "f1_binary": f1_bin, "f1_macro": f1_mac, "pr_auc": pr_auc}


class WeightedTrainer(Trainer):
    def __init__(self, loss_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_weight = loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = (model.module.config.num_labels if hasattr(model, "module") else model.config.num_labels)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.loss_weight.to(logits.device))
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True, collate_fn=default_data_collator)


def sklearn_class_weights(y_np: np.ndarray, device: str = "cpu") -> torch.Tensor:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_np)
    return torch.tensor(weights, dtype=torch.float, device=device)


def parse_list_int(s):
    return [int(x) for x in s.split(",") if x.strip()!=""]


def parse_list_str(s):
    return [x.strip() for x in s.split(",") if x.strip()!=""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_username", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default="")
    ap.add_argument("--model_name", type=str, default="bert-base-cased")
    ap.add_argument("--datasets", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="sentence")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=500)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--out_json", type=str, default="all_metrics_results.json")
    args = ap.parse_args()

    if args.push_to_hub and args.hf_token:
        login(token=args.hf_token)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    datasets = parse_list_str(args.datasets)
    seeds = parse_list_int(args.seeds)
    device = DEVICE
    all_metrics_results = {}

    for dataset in datasets:
        all_metrics_results[dataset] = {}
        train_path = f"{dataset}_train.csv"
        tr_texts, tr_labels = load_split(train_path, args.text_col, args.label_col)
        weights = sklearn_class_weights(np.array(tr_labels), device=device)
        for seed in seeds:
            set_seed(seed)
            lr_key = f"{args.lr:.0e}"
            all_metrics_results[dataset].setdefault(lr_key, {})
            all_metrics_results[dataset][lr_key].setdefault(f"seed_{seed}", {})
            train_ds = BinaryHateDataset(tr_texts, tr_labels, tokenizer, max_len=args.max_len)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=2,
                id2label={0: "not_hate", 1: "hate"},
                label2id={"not_hate": 0, "hate": 1},
            ).to(device)
            out_dir = Path(f"./tmp/{dataset}/seed{seed}_lr{lr_key}")
            args_tr = TrainingArguments(
                output_dir=str(out_dir),
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=seed,
                logging_strategy="no",
                save_strategy="no",
                report_to="none",
            )
            trainer = WeightedTrainer(
                loss_weight=weights,
                model=model,
                args=args_tr,
                train_dataset=train_ds,
                compute_metrics=compute_metrics,
                data_collator=default_data_collator,
                tokenizer=tokenizer,
            )
            trainer.train()
            base_model_name = args.model_name.split("/")[-1]
            save_dir = Path(f"{base_model_name}-{dataset}-s{seed}")
            save_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            for dataset_eval in datasets:
                eval_path = f"{dataset_eval}_test.csv"
                te_texts, te_labels = load_split(eval_path, args.text_col, args.label_col)
                test_ds = BinaryHateDataset(te_texts, te_labels, tokenizer, max_len=args.max_len)
                res = trainer.evaluate(eval_dataset=test_ds)
                all_metrics_results[dataset][lr_key][f"seed_{seed}"][dataset_eval] = {
                    "accuracy": float(res.get("eval_accuracy", np.nan)),
                    "f1_binary": float(res.get("eval_f1_binary", np.nan)),
                    "f1_macro": float(res.get("eval_f1_macro", np.nan)),
                    "pr_auc": float(res.get("eval_pr_auc", np.nan)),
                }
            if args.push_to_hub:
                repo_id = f"{args.hf_username}/{base_model_name}-{dataset}-s{seed}"
                api = HfApi(); api.create_repo(repo_id, exist_ok=True)
                model_to_push = AutoModelForSequenceClassification.from_pretrained(str(save_dir))
                tok_to_push   = AutoTokenizer.from_pretrained(str(save_dir))
                model_to_push.push_to_hub(repo_id)
                tok_to_push.push_to_hub(repo_id)

    with open(args.out_json, "w") as f:
        json.dump(all_metrics_results, f, indent=2)


if __name__ == "__main__":
    main()