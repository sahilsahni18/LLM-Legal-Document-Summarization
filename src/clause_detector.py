# src/clause_detector.py
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

MODEL_NAME = "bert-base-uncased"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return TOKENIZER(batch["text"], truncation=True, padding="max_length", max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train(data_csv: str, output_dir: str):
    ds = load_dataset("csv", data_files={"train": data_csv, "test": data_csv})
    ds = ds.map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    TOKENIZER.save_pretrained(output_dir)

def inference(texts: List[str], model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    preds = torch.softmax(outputs.logits, dim=-1)[:,1]
    return preds.tolist()

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train","infer"])
    parser.add_argument("--data", help="CSV for train/test")
    parser.add_argument("--model_dir", default="clause_model")
    parser.add_argument("--texts", help="JSON list of texts for inference")
    args = parser.parse_args()

    if args.mode=="train":
        train(args.data, args.model_dir)
    else:
        texts = json.loads(args.texts)
        scores = inference(texts, args.model_dir)
        print(scores)
