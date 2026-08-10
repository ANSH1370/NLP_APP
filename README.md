# 🧠 NLP Studio

### Interactive Natural Language Processing & Text Analytics Platform

**NLP Studio** is an interactive web-based Natural Language Processing platform that brings multiple NLP capabilities together into a single application.

The platform allows users to submit text and perform tasks such as **sentiment analysis, Named Entity Recognition (NER), Part-of-Speech (POS) tagging, and text preprocessing** through a simple web interface.

The application combines **machine-learning models, NLTK, spaCy, Scikit-learn, TF-IDF vectorization, and Flask** to provide an end-to-end NLP experimentation and analysis environment.

---

## ✨ Features

### 😊 Sentiment Analysis

Analyze the sentiment of user-provided text and classify it as:

* **Positive**
* **Negative**

The sentiment pipeline performs text preprocessing before passing the transformed text through a trained TF-IDF vectorizer and machine-learning classification model.

### 🔤 Part-of-Speech Tagging

Analyze the grammatical structure of a sentence by identifying the role of each word.

The application uses **spaCy's `en_core_web_sm` model** to generate POS tags and provides an explanation for the assigned tag.

Example:

```text
Input:
Ansh is developing an AI application.

Output:
Ansh        → Proper Noun
is          → Verb
developing  → Verb
an          → Determiner
AI          → Noun
application → Noun
```

---

### 🏷️ Named Entity Recognition

Extract named entities from text and identify their entity types.

The NER pipeline uses spaCy's English language model to detect entities from the supplied text.

Example:

```text
Input:
Apple opened a new office in Mumbai.

Entities:

Apple   → Organization
Mumbai  → Geopolitical Entity
```

---

### 🧹 Text Preprocessing

The application includes preprocessing techniques used by the sentiment-analysis pipeline, including:

* Tokenization
* Stop-word removal
* Stemming
* Text normalization

The sentiment workflow uses NLTK's tokenizer, English stop-word corpus, and Porter Stemmer before TF-IDF transformation.

---

### 🤖 Machine-Learning-Based Sentiment Classification

Unlike the spaCy-based linguistic tasks, sentiment analysis uses a separately trained machine-learning pipeline.

The repository contains:

```text
Newest1_NLP_Model.pkl
Newest1_TfidVectorizer_2.pkl
```

The Flask application loads these serialized artifacts and uses them during inference.

---

### 🔐 User Authentication

The application also includes a basic registration and login workflow.

Users can:

* Register an account
* Log in
* Maintain a session
* Access the NLP application after authentication

The Flask application manages authentication state using Flask sessions and connects to a database layer through `mydb.py`.

---

# 🏗️ System Architecture

```text
                         ┌──────────────────┐
                         │       User       │
                         └────────┬─────────┘
                                  │
                                  ▼
                       ┌────────────────────┐
                       │   Web Interface    │
                       │ HTML + Jinja       │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │   Flask Backend    │
                       └─────────┬──────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │  Sentiment   │ │     POS      │ │     NER      │
        │   Analysis   │ │   Tagging    │ │ Recognition  │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               ▼                ▼                ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ TF-IDF + ML  │ │    spaCy     │ │    spaCy     │
        │    Model     │ │ NLP Pipeline │ │ NLP Pipeline │
        └──────────────┘ └──────────────┘ └──────────────┘
```

---

# 🔄 NLP Processing Pipeline

## Sentiment Analysis

```text
User Input
    │
    ▼
Tokenization
    │
    ▼
Stop-word Removal
    │
    ▼
Stemming
    │
    ▼
TF-IDF Vectorization
    │
    ▼
Trained ML Model
    │
    ▼
Positive / Negative
```

The implementation loads the trained TF-IDF vectorizer and classification model from serialized `.pkl` files and performs prediction on the transformed input.

---

## POS Tagging

```text
User Input
    │
    ▼
spaCy NLP Pipeline
    │
    ▼
Tokenization
    │
    ▼
POS Tagging
    │
    ▼
Tag Explanation
    │
    ▼
Formatted Results
```

The application loads `en_core_web_sm`, processes the document, and returns each token along with an explanation of its POS tag.

---

## Named Entity Recognition

```text
User Input
    │
    ▼
spaCy NLP Pipeline
    │
    ▼
Entity Detection
    │
    ▼
Entity Classification
    │
    ▼
Entity + Label
```

The NER implementation iterates over `doc.ents` and returns detected entities along with explanations of their labels.

---

# 🧠 NLP Components

The project brings together several fundamental NLP concepts:

| Component                | Technology         | Purpose                              |
| ------------------------ | ------------------ | ------------------------------------ |
| Text Tokenization        | NLTK               | Split text into tokens               |
| Stop-word Removal        | NLTK               | Remove common words                  |
| Stemming                 | Porter Stemmer     | Reduce words to stems                |
| TF-IDF                   | Scikit-learn       | Convert text into numerical features |
| Sentiment Classification | Scikit-learn model | Predict sentiment                    |
| POS Tagging              | spaCy              | Identify grammatical roles           |
| NER                      | spaCy              | Identify named entities              |

The repository's requirements include spaCy, NumPy, Pandas, Scikit-learn, Flask, Jinja2, Gunicorn, Gensim, and NLTK.

---

# 📊 Sentiment Analysis Model

The sentiment-analysis pipeline follows a traditional supervised machine-learning approach:

```text
                    Training Data
                         │
                         ▼
                 Text Preprocessing
                         │
                         ▼
                  TF-IDF Vectorizer
                         │
                         ▼
                 Feature Representation
                         │
                         ▼
                  ML Classification
                         │
                         ▼
                 Serialized Model
                         │
                         ▼
                  Flask Application
                         │
                         ▼
                     Prediction
```

During inference, the application:

1. Receives the user's text.
2. Removes English stop words.
3. Applies Porter stemming.
4. Transforms the processed text using the saved TF-IDF vectorizer.
5. Passes the resulting feature vector to the trained model.
6. Returns **Positive** or **Negative**.

---

# 📚 Development Notebooks

The repository contains dedicated notebooks for different NLP components:

```text
Name Entity Recognition.ipynb
        │
        └── NER experimentation

Parts of Speech Tagging.ipynb
        │
        └── POS tagging experimentation

Text_Preparation And Cleaning .ipynb
        │
        └── Text preprocessing

Text_Sentiment_Modeling.ipynb
        │
        └── Sentiment model development
```

This separation makes it possible to experiment with individual NLP techniques before integrating them into the web application.

---

# 🛠️ Technology Stack

### Backend

* **Python**
* **Flask**
* **Jinja2**
* **Gunicorn**

### NLP

* **NLTK**
* **spaCy**
* **Gensim**

### Machine Learning

* **Scikit-learn**
* **TF-IDF**
* Serialized ML models using **Pickle**

### Data Processing

* **NumPy**
* **Pandas**

### Frontend

* HTML
* CSS
* Jinja templates

The technologies above are reflected in the repository's current dependency list and application code.

---

# 📂 Project Structure

```text
NLP_APP/
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── sentiment.html
│   ├── POS.html
│   ├── POS_output.html
│   ├── NER.html
│   ├── NER_output.html
│   └── ...
│
├── Name Entity Recognition.ipynb
├── Parts of Speech Tagging.ipynb
├── Text_Preparation And Cleaning .ipynb
├── Text_Sentiment_Modeling.ipynb
│
├── Text_Sentiment.py
├── app.py
├── mydb.py
│
├── Newest1_NLP_Model.pkl
├── Newest1_TfidVectorizer_2.pkl
│
├── requirements.txt
├── users.json
└── temp.py
```

The current repository contains the Flask application, NLP notebooks, serialized sentiment artifacts, templates, database helper, and dependency file.

---

# ⚙️ Installation

## Prerequisites

Make sure you have:

* Python 3.x
* pip
* Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/ANSH1370/NLP_APP.git

cd NLP_APP
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The current requirements file includes the project's NLP, machine-learning, and Flask dependencies.

---

## 4. Install the spaCy English Model

The application uses:

```text
en_core_web_sm
```

Install it with:

```bash
python -m spacy download en_core_web_sm
```

The application also contains fallback logic to download the model if it is unavailable at runtime.

---

## 5. Download NLTK Resources

The application requires NLTK resources such as:

* `punkt`
* `stopwords`

The application contains runtime checks for these resources.

You can also install them manually:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
```

---

# ▶️ Running the Application

Run the Flask application:

```bash
python app.py
```

The application is configured to serve through:

```text
http://localhost:8080
```

The repository's `app.py` uses **Waitress** to serve the application on `0.0.0.0:8080`.

Open the application in your browser:

```text
http://localhost:8080
```

---

# 🧪 Example Use Cases

## Sentiment Analysis

```text
Input:
I really enjoyed using this product. The experience was excellent!

Output:
Positive
```

```text
Input:
The service was disappointing and the product did not work properly.

Output:
Negative
```

---

## POS Tagging

```text
Input:
The developer built an intelligent application.

Output:

The          → Determiner
developer    → Noun
built        → Verb
an           → Determiner
intelligent  → Adjective
application  → Noun
```

---

## NER

```text
Input:
Google opened a new research center in Bengaluru.

Output:

Google      → Organization
Bengaluru   → Geopolitical Entity
```

---

# 🎯 Learning Objectives

This project demonstrates practical implementation of several foundational NLP concepts:

* Text preprocessing
* Tokenization
* Stop-word removal
* Stemming
* Feature extraction using TF-IDF
* Supervised text classification
* Sentiment analysis
* Named Entity Recognition
* Part-of-Speech tagging
* NLP model integration
* Model serialization
* Flask application development
* Session-based authentication
* Deployable Python web application architecture

---

# 🔮 Future Improvements

The project can be extended into a more comprehensive NLP platform by adding:

* [ ] Neutral sentiment classification
* [ ] Emotion detection
* [ ] Text summarization
* [ ] Keyword extraction
* [ ] Text similarity
* [ ] Language detection
* [ ] Spam detection
* [ ] Question answering
* [ ] Topic classification
* [ ] Word-cloud visualization
* [ ] Text statistics dashboard
* [ ] Transformer-based NLP models
* [ ] BERT-based sentiment analysis
* [ ] Multilingual NLP
* [ ] REST API endpoints
* [ ] Docker deployment
* [ ] Modern frontend
* [ ] Improved authentication and password security

---

# 🚀 Future Vision

The project can evolve from a collection of individual NLP utilities into a unified **Text Intelligence Platform**:

```text
                         ┌─────────────────────┐
                         │     Input Text      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   NLP Processing    │
                         └──────────┬──────────┘
                                    │
            ┌───────────────┬───────┼────────┬───────────────┐
            ▼               ▼       ▼        ▼               ▼
        Sentiment          NER     POS     Keywords      Similarity
            │               │       │        │               │
            └───────────────┴───────┼────────┴───────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Unified Insights   │
                         └─────────────────────┘
```

---

# ⚠️ Security & Production Considerations

This project was primarily developed as an NLP learning and application-development project.

Before production deployment, several areas should be improved:

* Use secure password hashing.
* Store secrets in environment variables.
* Avoid committing user data or credentials.
* Add CSRF protection.
* Add proper input validation.
* Pin dependency versions.
* Separate model-loading logic from application routes.
* Add automated tests.
* Add structured logging.
* Use a production WSGI configuration.
* Add API authentication if exposing inference endpoints.

---

# 👨‍💻 Author

**Ansh Mangukiya**

AI Engineer | Machine Learning | NLP | Generative AI

GitHub:
https://github.com/ANSH1370

---

# ⭐ Project

If you find this project useful or interesting, feel free to explore the implementation and give the repository a ⭐.

**Repository:**
https://github.com/ANSH1370/NLP_APP
