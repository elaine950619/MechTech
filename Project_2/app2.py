import time
import psutil
import requests
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Start timing the entire pipeline
pipeline_start = time.time()
process = psutil.Process()

# --- Data Ingestion ---
ingestion_start = time.time()
mem_before_ingestion = process.memory_info().rss

API_KEY = os.environ["API_KEY"]
url = f'https://newsapi.org/v2/top-headlines?category=business&apiKey={API_KEY}'
response = requests.get(url)
articles = response.json().get('articles', [])
logging.info("Fetched articles from API.")

# Extract titles and publication dates
news_data = []
for article in articles:
    news_data.append({
        'title': article.get('title'),
        'publishedAt': article.get('publishedAt')
    })
df_news = pd.DataFrame(news_data)

mem_after_ingestion = process.memory_info().rss
ingestion_end = time.time()

logging.info("Data Ingestion completed in %.2f seconds", ingestion_end - ingestion_start)
logging.info("Memory increased by %.2f MB during ingestion", (mem_after_ingestion - mem_before_ingestion) / (1024*1024))

# --- Sentiment Analysis ---
sentiment_start = time.time()
mem_before_sentiment = process.memory_info().rss

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(title):
    return analyzer.polarity_scores(title)

df_news['sentiment'] = df_news['title'].apply(get_sentiment)
df_news = df_news.join(df_news['sentiment'].apply(pd.Series))
df_news.drop(columns=['sentiment'], inplace=True)

mem_after_sentiment = process.memory_info().rss
sentiment_end = time.time()

logging.info("Sentiment Analysis completed in %.2f seconds", sentiment_end - sentiment_start)
logging.info("Memory increased by %.2f MB during sentiment analysis", (mem_after_sentiment - mem_before_sentiment) / (1024*1024))

# --- Database Insertion ---
db_insertion_start = time.time()
mem_before_db = process.memory_info().rss

conn = psycopg2.connect(
    dbname='news_sentiment_db',
    user='elaine',
    password='password',
    host='localhost',
    port='5432'
)
cur = conn.cursor()

insert_query = """
INSERT INTO news (title, published_at, sentiment_positive, sentiment_negative, sentiment_neutral, sentiment_compound)
VALUES (%s, %s, %s, %s, %s, %s);
"""

for index, row in df_news.iterrows():
    published_at = row['publishedAt']
    try:
        if published_at and published_at.endswith('Z'):
            published_at = published_at[:-1]
        published_at = datetime.fromisoformat(published_at)
    except Exception:
        published_at = None

    cur.execute(insert_query, (
        row['title'],
        published_at,
        row.get('pos', 0),
        row.get('neg', 0),
        row.get('neu', 0),
        row.get('compound', 0)
    ))

conn.commit()
cur.close()
conn.close()

mem_after_db = process.memory_info().rss
db_insertion_end = time.time()

logging.info("Database Insertion completed in %.2f seconds", db_insertion_end - db_insertion_start)
logging.info("Memory increased by %.2f MB during DB insertion", (mem_after_db - mem_before_db) / (1024*1024))

pipeline_end = time.time()
logging.info("Total pipeline runtime: %.2f seconds", pipeline_end - pipeline_start)

# Throughput Example:
num_articles = len(df_news)
throughput = num_articles / (pipeline_end - pipeline_start)
logging.info("Processed %d articles, throughput: %.2f articles/second", num_articles, throughput)
