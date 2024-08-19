from flask import Flask

app = Flask(__name__)

@app.route('/')

def index():
    return 'Hello Ansh'

app.run(debug=True) # Use to reload and run