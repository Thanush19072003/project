from flask import Flask, render_template, request
import sqlite3 as sql
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('login.html')

@app.route('/enternew')
def new_student():
   return render_template('register.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
   if request.method == 'POST':
      try:
         nm = request.form['USER ID']
         addr = request.form['email']
         pin = request.form['password']
         
         with sql.connect("database.db") as con:
            cur = con.cursor()
            
            cur.execute("INSERT INTO students (name,email,password)VALUES (?,?,?,?)",(nm,addr,pin) )
            
            con.commit()
            msg = "Record successfully added"
      except:
         con.rollback()
         msg = "error in insert operation"
      
      finally:
         return render_template("register.html",msg = msg)
         con.close()

@app.route('/list')
def list():
   con = sql.connect("medical.db")
   con.row_factory = sql.Row
   
   cur = con.cursor()
   cur.execute("select * from students")
   
   rows = cur.fetchall();
   return render_template("register.html",rows = rows)

if __name__ == '__main__':
   app.run(debug = True)