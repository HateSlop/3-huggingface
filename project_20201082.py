# !pip install -q transformers datasets torch matplotlib
import torch
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

device = 0 if torch.cuda.is_available() else -1
dataset = load_dataset('imdb', split='test')
dataset = dataset.select(range(100))
print(dataset)
sentiment_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=device)
def analyze_sentiment(batch):
    results = sentiment_model(batch['text'], truncation=True, max_length=512)
    batch['predicted_label'] = [r['label'] for r in results]
    batch['confidence'] = [r['score'] for r in results]
    return batch

dataset = dataset.map(analyze_sentiment, batched=True, batch_size=8)
df = pd.DataFrame(dataset)
print(df[['text', 'label', 'predicted_label', 'confidence']].head(10))
correct = sum(1 for i in range(len(df)) if (df['label'][i] == 1 and df['predicted_label'][i] == 'POSITIVE') or (df['label'][i] == 0 and df['predicted_label'][i] == 'NEGATIVE'))
accuracy = correct / len(df)
print(f'Accuracy: {accuracy:.2%}')

positive_count = sum(1 for label in df['predicted_label'] if label == 'POSITIVE')
negative_count = sum(1 for label in df['predicted_label'] if label == 'NEGATIVE')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(['POSITIVE', 'NEGATIVE'], [positive_count, negative_count], color=['green', 'red'])
axes[0].set_title('Predicted Sentiment Distribution')
axes[0].set_ylabel('Count')

axes[1].hist(df['confidence'], bins=20, color='blue', alpha=0.7, edgecolor='black')
axes[1].set_title('Confidence Score Distribution')
axes[1].set_xlabel('Confidence')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()