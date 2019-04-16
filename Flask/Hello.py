

'''from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello Hello'
    
if __name__ == '__main__':
    app.run(debug=True)'''
    
    
'''@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route('/rev/<float:revNo>')
def revision(revNo):
   return 'Revision Number %f' % revNo

if __name__ == '__main__':
   app.run()  '''




from flask import Flask, render_template, url_for,request,redirect
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('hello.html')
    
@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':      
        user = request.form['nm']
        return redirect(url_for('success',name = user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success',name = user))
    
if __name__ == '__main__':
    app.run()
