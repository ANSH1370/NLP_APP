from flask import Flask,render_template,request,redirect,session,g
from mydb import Database
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from nltk import sent_tokenize
from nltk.stem.porter import PorterStemmer
import spacy
import os

app = Flask(__name__)
db_obj = Database()

app.secret_key = os.urandom(24)

tfidf = pickle.load(open('Newest_TfidVectorizer_2.pkl', 'rb'))
model = pickle.load(open('Newest_NLP_Model.pkl', 'rb'))

@app.route('/')
def main():
    session.pop('user',None)
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/perform_registration',methods=['post'])
def perform_registration():
    name = request.form.get('uname')
    email = request.form.get('uemail')
    password = request.form.get('upass')

    response = db_obj.insert(name,email,password)

    if response:
        return render_template('login.html' ,message='Registration Successful,Kindly Login to Proceed')

    else:
        return render_template('Register.html',message='Email Already Exists!')

@app.route('/perform_login',methods=['post'])
def perform_login():
    email = request.form.get('uemail')
    password = request.form.get('upass')

    response = db_obj.search(email,password)

    if response:
        session['user'] = email
        return redirect('/Home')
    else:
        return render_template('login.html',message='Incorrect Email or Password!')
@app.route('/Home')
def home():
    if g.user:
        return render_template('home.html',user = session['user'])
    return render_template('login.html')

@app.before_request
def before_request():
    g.user = None
    if 'user' in session:
        g.user = session['user']

@app.route('/text_sentiment_analysis')
def text_sentiment_analysis():
    return render_template('sentiment.html')

@app.route('/sentimentation',methods=['post'])
def sentimentation():
    text = request.form.get('text')
    sw = stopwords.words('english')
    ps = PorterStemmer()

    def remove_stopwords(text):
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if word.lower() not in sw]
        return ' '.join(filtered_text)

    def stem_text(text):
        stemed_list = []
        tokens = word_tokenize(text)
        for i in tokens:
            stemed_list.append(ps.stem(i))
        return ' '.join(stemed_list)
    text = remove_stopwords(text)
    text = stem_text(text)
    text = [text]

    vector_inputs = tfidf.transform(text)
    result = model.predict(vector_inputs)
    if result[0] == 0:
        return render_template('sentiment.html',message='Negative')
    else:
        return render_template('sentiment.html',message='Positive')

@app.route('/POS_tagging')
def POS_tagging():
    return render_template('POS.html')

@app.route('/tagging',methods=['post'])
def tagging():
    text = request.form.get('text')
    result = ''
    l=[]
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    for word in doc:
        l.append((word,spacy.explain(word.tag_)))
    return render_template('POS_output.html',pos_tags = l)

@app.route('/ner')
def ner():
    return render_template('NER.html')

@app.route('/ner_results',methods = ['post'])
def ner_results():
    text = request.form.get('text')
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    l=[]
    for ent in doc.ents:
        l.append((ent,spacy.explain(ent.label_)))
    return render_template('NER_output.html',ner_tags = l)

app.run(debug=True)
