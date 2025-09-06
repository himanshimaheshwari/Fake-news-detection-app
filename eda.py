import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Handle missing or empty cleaned_text
df['cleaned_text'] = df['cleaned_text'].fillna('')  # Replace NaN with empty string
df['cleaned_text'] = df['cleaned_text'].astype(str)  # Ensure string type

# Article lengths (skip empty strings)
df['length'] = df['cleaned_text'].apply(lambda x: len(x) if x else 0)

# Plot distribution of article lengths (exclude zero-length)
plt.figure(figsize=(10, 6))
sns.histplot(df[(df['label'] == 1) & (df['length'] > 0)]['length'], color='blue', label='Real', kde=True)
sns.histplot(df[(df['label'] == 0) & (df['length'] > 0)]['length'], color='red', label='Fake', kde=True)
plt.title('Distribution of Article Lengths')
plt.xlabel('Length')
plt.legend()
plt.savefig('article_lengths.png')
plt.close()

# Word clouds (only for non-empty text)
real_text = ' '.join(df[df['label'] == 1]['cleaned_text'])
fake_text = ' '.join(df[df['label'] == 0]['cleaned_text'])

# Real news word cloud
if real_text.strip():  # Check if non-empty
    wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
    wordcloud_real.to_file('wordcloud_real.png')
else:
    print("No valid text for real news word cloud.")

# Fake news word cloud
if fake_text.strip():  # Check if non-empty
    wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
    wordcloud_fake.to_file('wordcloud_fake.png')
else:
    print("No valid text for fake news word cloud.")

print("EDA done. Images saved: article_lengths.png, wordcloud_real.png, wordcloud_fake.png")
