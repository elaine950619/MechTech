import sqlite3
from werkzeug.security import generate_password_hash

connection = sqlite3.connect('database.db')

with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute(
    "INSERT INTO users (username, password) VALUES (?, ?)",
    ('admin', generate_password_hash('admin123'))
)

connection.commit()
connection.close()