
import nltk

nltk_data_path = '/opt/render/nltk_data'
nltk.data.path.append(nltk_data_path)

# Ensure the 'punkt' tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    

# # Additional setup code for stopwords
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords', download_dir=nltk_data_path)
#     sw = stopwords.words('english')

from flask import Flask, render_template, request, redirect, session, g
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

tfidf = pickle.load(open('Newest1_TfidVectorizer_2.pkl', 'rb'))
model = pickle.load(open('Newest1_NLP_Model.pkl', 'rb'))

@app.route('/')
def main():
    session.pop('user', None)
    return render_template('home.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/perform_registration', methods=['POST'])
def perform_registration():
    name = request.form.get('uname')
    email = request.form.get('uemail')
    password = request.form.get('upass')

    response = db_obj.insert(name, email, password)

    if response:
        return render_template('login.html', message='Registration Successful, Kindly Login to Proceed')
    else:
        return render_template('register.html', message='Email Already Exists!')

@app.route('/perform_login', methods=['POST'])
def perform_login():
    email = request.form.get('uemail')
    password = request.form.get('upass')

    response = db_obj.search(email, password)

    if response:
        session['user'] = email
        return redirect('/Home')
    else:
        return render_template('login.html', message='Incorrect Email or Password!')

@app.route('/Home')
def home():
    if g.user:
        return render_template('home.html', user=session['user'])
    return render_template('login.html')

@app.before_request
def before_request():
    g.user = None
    if 'user' in session:
        g.user = session['user']

@app.route('/text_sentiment_analysis')
def text_sentiment_analysis():
    return render_template('sentiment.html')

@app.route('/sentimentation', methods=['POST'])
def sentimentation():
    text = request.form.get('text')
    try:
        sw = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        sw = stopwords.words('english')
    ps = PorterStemmer()
    # ps = PorterStemmer()

    def remove_stopwords(text):
        tokens = word_tokenize(text)
        filtered_text = [word for word in tokens if word.lower() not in sw]
        return ' '.join(filtered_text)

    def stem_text(text):
        stemmed_list = []
        tokens = word_tokenize(text)
        for i in tokens:
            stemmed_list.append(ps.stem(i))
        return ' '.join(stemmed_list)

    try:
        text = remove_stopwords(text)
        text = stem_text(text)
        text = [text]
        vector_inputs = tfidf.transform(text)
        result = model.predict(vector_inputs)
    except Exception as e:
        print(f"Error in text processing or model prediction: {e}")
        return render_template('sentiment.html', message='An error occurred during processing.')

    if result[0] == 0:
        return render_template('sentiment.html', message='Negative')
    else:
        return render_template('sentiment.html', message='Positive')

@app.route('/POS_tagging')
def POS_tagging():
    return render_template('POS.html')

@app.route('/tagging', methods=['POST'])
def tagging():
    text = request.form.get('text')
    result = ''
    l = []

    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')

    doc = nlp(text)
    for word in doc:
        l.append((word, spacy.explain(word.tag_)))
    return render_template('POS_output.html', pos_tags=l)

@app.route('/ner')
def ner():
    return render_template('NER.html')

@app.route('/ner_results', methods=['POST'])
def ner_results():
    try:
        text = request.form.get('text')
        if not text:
            raise ValueError("No text provided for NER.")

        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')

        doc = nlp(text)
        l = []
        for ent in doc.ents:
            l.append((ent, spacy.explain(ent.label_)))

        return render_template('NER_output.html', ner_tags=l)
    except Exception as e:
        app.logger.error(f"Error in NER processing: {e}")
        return render_template('NER.html', error="An error occurred during NER processing. Please try again.")

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
