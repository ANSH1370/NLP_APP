import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from nltk import sent_tokenize
from nltk.stem.porter import PorterStemmer

sw = stopwords.words('english')
ps = PorterStemmer()

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_text = [word for word in tokens if word.lower() not in sw]
    return ' '.join(filtered_text)

def stem_text(text):
    stemed_list=[]
    tokens = word_tokenize(text)
    for i in tokens:
        stemed_list.append(ps.stem(i))
    return ' '.join(stemed_list)

tfidf = pickle.load(open('TfidVectorizer.pkl','rb'))
model = pickle.load(open('Model.pkl','rb'))

