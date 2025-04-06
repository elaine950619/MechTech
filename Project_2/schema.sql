DROP TABLE IF EXISTS news;

CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    title TEXT,
    published_at TIMESTAMP,
    sentiment_positive REAL,
    sentiment_negative REAL,
    sentiment_neutral REAL,
    sentiment_compound REAL
); 

DROP TABLE IF EXISTS aggregated_data;
CREATE TABLE aggregated_data (
    id SERIAL PRIMARY KEY,
    published_date DATE,
    avg_sentiment_positive REAL,
    avg_sentiment_negative REAL,
    avg_sentiment_neutral REAL,
    avg_sentiment_compound REAL,
    closing_price REAL
);