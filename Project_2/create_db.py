import psycopg2

conn = psycopg2.connect(
    dbname='news_sentiment_db',
    user='elaine',
    password='password',
    host='localhost',
    port='5432'
)
cur = conn.cursor()

# Read schema.sql and execute
with open('schema.sql', 'r') as f:
    schema_sql = f.read()
cur.execute(schema_sql)
conn.commit()

print("Database table 'news' created successfully!")

cur.close()
conn.close()