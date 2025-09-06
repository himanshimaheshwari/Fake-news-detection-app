
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Check file existence
for file in ['X_test.pkl', 'y_test.pkl', 'lr_model.pkl', 'rf_model.pkl', 'nb_model.pkl', 'test_data.csv', 'bert_model']:
    if not os.path.exists(file):
        print(f"Error: {file} not found. Run train_basic.py or train_bert.py first.")
        exit()

# Load test data for basic models
X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')

# Basic models
models = {
    'Logistic Regression': joblib.load('lr_model.pkl'),
    'Random Forest': joblib.load('rf_model.pkl'),
    'Naive Bayes': joblib.load('nb_model.pkl')
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"{name} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Fix f-string syntax
    filename = f'cm_{name.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    plt.close()

# BERT evaluation
try:
    model = BertForSequenceClassification.from_pretrained('bert_model')
    tokenizer = BertTokenizer.from_pretrained('bert_model')
except Exception as e:
    print(f"Error loading BERT model: {str(e)}")
    exit()

test_df = pd.read_csv('test_data.csv')

# Handle missing or empty cleaned_text
test_df['cleaned_text'] = test_df['cleaned_text'].fillna('').astype(str)

# Tokenize test
test_encodings = tokenizer(test_df['cleaned_text'].tolist(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')

# Dataset
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = NewsDataset(test_encodings, test_df['label'].values)

# Trainer for prediction
trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label=1)
rec = recall_score(y_true, y_pred, pos_label=1)
f1 = f1_score(y_true, y_pred, pos_label=1)
print(f"BERT - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix - BERT')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('cm_bert.png')
plt.close()

print("Evaluation complete. Metrics printed in console. Confusion matrices saved as cm_*.png.")
