import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load data (use full dataset)
df = pd.read_csv('preprocessed_data.csv')  
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save test_df for evaluation
test_df.to_csv('test_data.csv', index=False)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['cleaned_text'].tolist(), padding='max_length', truncation=True, max_length=128)

train_encodings = tokenize_function(train_df)
test_encodings = tokenize_function(test_df)

# Dataset class
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.int64)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_df['label'].values)
test_dataset = NewsDataset(test_encodings, test_df['label'].values)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir='./bert_model',
    num_train_epochs=3,  
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Save
model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model')

print("BERT model trained and saved in 'bert_model' folder.")
