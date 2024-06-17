from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

app = Flask(__name__)
app.secret_key = 'chatbot'

# Database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'asc'
app.config['MYSQL_DB'] = 'validate'

mysql = MySQL(app)
@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/')
@app.route('/about')
def about():
	return render_template('about.html')


@app.route('/')
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/loginaction', methods=['POST'])
def loginaction():
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['pwd']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM signin WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        if account:
            return 'hi'
        else:
            flash('Invalid Login')
            return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        reg = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{6,10}$"
        pattern = re.compile(reg)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM signin WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not re.search(pattern, password):
            msg = 'Password should contain at least one number, one lower case character, one uppercase character, one special symbol and must be between 6 to 10 characters long'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO signin VALUES (NULL, %s, %s, %s, "Approved")', (username, email, password))
            mysql.connection.commit()
            flash('You have successfully registered! Please proceed to login.')
            return redirect(url_for('login'))
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('register.html', msg=msg)

if __name__ == '__main__':
  app.run(debug=True)