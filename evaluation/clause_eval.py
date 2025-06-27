import pandas as pd
from src.clause_detector import inference
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv("data/labeled_clauses.csv")  # columns text,label
scores = inference(df["text"].tolist(), model_dir="clause_model")
preds = [1 if s>0.5 else 0 for s in scores]

acc = accuracy_score(df["label"], preds)
prec, rec, f1, _ = precision_recall_fscore_support(df["label"], preds, average='binary')

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
