import requests
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Get news data
API_KEY = os.environ["API_KEY"]
url = f'https://newsapi.org/v2/top-headlines?category=business&apiKey={API_KEY}'
response = requests.get(url)
print(os.environ["API_KEY"])
print(response.json())
articles = response.json().get('articles', [])

# Extract titles and publication dates
news_data = []
for article in articles:
    news_data.append({
        'title': article.get('title'),
        'publishedAt': article.get('publishedAt')
    })

df_news = pd.DataFrame(news_data)
print(df_news.head())


analyzer = SentimentIntensityAnalyzer()
# Example: analyze sentiment of a news headline
# headline = "Market surges amid economic optimism"
# sentiment = analyzer.polarity_scores(headline)
# print(sentiment)

# Analyze sentiment of all news headlines
def get_sentiment(title):
    return analyzer.polarity_scores(title)

df_news['sentiment'] = df_news['title'].apply(get_sentiment)

# Expand sentiment scores into separate columns
df_news = df_news.join(df_news['sentiment'].apply(pd.Series))
df_news.drop(columns=['sentiment'], inplace=True)
print(df_news.head())

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname='news_sentiment_db',
    user='elaine',        # Adjust if needed
    password='password',  # Adjust if needed
    host='localhost',
    port='5432'
)
cur = conn.cursor()

insert_query = """
INSERT INTO news (title, published_at, sentiment_positive, sentiment_negative, sentiment_neutral, sentiment_compound)
VALUES (%s, %s, %s, %s, %s, %s);
"""

for index, row in df_news.iterrows():
    # Convert 'publishedAt' string to a datetime object if necessary
    published_at = row['publishedAt']

    try:
        # Some APIs return ISO 8601 strings; strip trailing 'Z' if present
        if published_at and published_at.endswith('Z'):
            published_at = published_at[:-1]
        published_at = datetime.fromisoformat(published_at)
    except Exception:
        # If parsing fails, you can set to None or handle differently
        published_at = None

    # Insert data into the database
    cur.execute(insert_query, (
        row['title'],
        published_at,
        row.get('pos', 0),
        row.get('neg', 0),
        row.get('neu', 0),
        row.get('compound', 0)
    ))

# Commit the transaction
conn.commit()
print("Data inserted successfully!")
# Close the cursor and connection
cur.close()
conn.close()