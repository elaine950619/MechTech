from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from backend.marketviews_sentiment_panel_finalized import get_data_to_draw, draw_sentiment_panel
import sqlite3

app = Flask(__name__)
app.secret_key = "mysecretkey"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

class User: 
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.passwordhash = password

    def check_password(self, raw_password):
        return check_password_hash(self.passwordhash, raw_password)
    
    def is_authenticated(self):
        return True
    
    def is_active(self):
        return True 
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)
    
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user_row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user_row is None:
        return None
    
    return User(user_row['id'], user_row['username'], user_row['password'])

def get_user_by_username(username):
    conn = get_db_connection()
    user_row = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return user_row

def create_user(username, password): 
    conn =  get_db_connection()
    conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, generate_password_hash(password)))
    conn.commit()
    conn.close()
    

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_row = get_user_by_username(username)
        if user_row:
            user = User(
                id=user_row['id'],
                username=user_row['username'],
                password=user_row['password']
            )
            if user and user.check_password(password):
                print('Login successful!')
                login_user(user)
                return redirect(url_for('dashboard'))
            
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register(): 
    if request.method == 'POST': 
        username = request.form['username']
        password = request.form['password']

        existing_user = get_user_by_username(username)
        if existing_user:
            flash('User already exists. ')
        else: 
            create_user(username, password)
            flash('Registration successful. Please login.')
            return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Call your data retrieval and treemap generation functions
    top5, low5 = get_data_to_draw(debug=True)
    print("=== TOP 5 EACH SECTOR ===")
    print(top5)
    print("=== LOW 5 EACH SECTOR ===")
    print(low5)
    positive_json, negative_json = draw_sentiment_panel(top5, low5)
    return render_template('dashboard.html', positive_plot=positive_json, negative_plot=negative_json)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
def index():
    # Optionally redirect root URL to /login
    return render_template('login.html')
    # or: return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)