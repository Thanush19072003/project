from flask import Flask, render_template, request
from flask import *
app = Flask(__name__)

from flask_mysqldb import MySQL
import MySQLdb.cursors
app = Flask(__name__) #Initialize the flask App
app.secret_key = 'chatbot'
app.config['MYSQL_HOST'] ='localhost'
app.config['MYSQL_USER'] ='root'
app.config['MYSQL_PASSWORD'] ='asc'
app.config['MYSQL_DB'] ='validate'
@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/')
@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html') 

@app.route('/loginaction', methods =['GET', 'POST'])
def loginaction():

    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['pwd']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM signin WHERE username = % s AND password = % s', (username, password))
        account = cursor.fetchone()
        if account:
            return 'hi'
        else:
            return 'Invalid Login'

@app.route('/register',methods= ['GET',"POST"])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        reg = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{6,10}$"
        pattern = re.compile(reg)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # Check if account exists using MySQL)
        cursor.execute('SELECT * FROM signin WHERE Username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not re.search(pattern,password):
            msg = 'Password should contain atleast one number, one lower case character, one uppercase character,one special symbol and must be between 6 to 10 characters long'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into employee table
            cursor.execute('INSERT INTO signin VALUES (NULL, %s, %s, %s, "Approved")', (username, email, password))
            mysql.connection.commit()
            flash('You have successfully registered! Please proceed for login!')
            return redirect(url_for('login'))
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
        return msg
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)





   
if __name__ == '__main__':
   app.run(debug = True)
