# Natural_language_processing_nlp
This README can be used for a repository that demonstrates these preprocessing steps on the provided sample text.

# NLP Preprocessing Pipeline

This repository provides a Python-based NLP preprocessing pipeline that covers the following steps:

* ✅ **Text Cleaning**
* ✅ **Tokenization**
* ✅ **Stopword Removal**
* ✅ **Stemming & Lemmatization**
* ✅ **Regex Preprocessing**
* ✅ **POS Tagging**
* ✅ **Named Entity Recognition (NER)**

The example text provided in this repository demonstrates the preprocessing steps for cleaning, tokenization, POS tagging, and more.

## Table of Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Preprocessing Steps](#preprocessing-steps)

  * [Text Cleaning](#text-cleaning)
  * [Tokenization](#tokenization)
  * [Stopword Removal](#stopword-removal)
  * [Stemming & Lemmatization](#stemming--lemmatization)
  * [Regex Preprocessing](#regex-preprocessing)
  * [POS Tagging](#pos-tagging)
  * [Named Entity Recognition (NER)](#named-entity-recognition-ner)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)

## Introduction

This repository contains an end-to-end NLP preprocessing pipeline. The goal is to prepare text data for downstream natural language processing tasks such as sentiment analysis, machine translation, or topic modeling. The pipeline handles the following tasks:

* Removing noise such as URLs and emojis
* Tokenizing the text
* Removing stopwords
* Lemmatizing and stemming words
* Recognizing named entities
* Annotating parts of speech

The following example paragraph demonstrates these preprocessing tasks.

### Sample Text:

"Hey! 👋 I just found this cool website 📱: [https://www.example.com](https://www.example.com) that has amazing articles on AI 🤖. I think @john_doe should check it out! It’s a great read, especially for tech enthusiasts like him. Have you seen the latest article by Dr. Alice Smith on the future of robotics? 🚀 I can't wait to learn more! 🔍 Also, don’t forget to follow their social media for updates 🌐."

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/nlp-preprocessing.git
   cd nlp-preprocessing
   ```

2. **Install dependencies**:
   This project requires the following Python libraries:

   * spaCy
   * NLTK
   * regex
   * pandas (optional)

   You can install them via pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model** (for POS tagging and NER):

   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Preprocessing Steps

### Text Cleaning

Remove unwanted characters such as emojis, URLs, and special characters.

**Example Code**:

```python
import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove emojis
    text = re.sub(r'[^\w\s,]', '', text)
    
    return text
```

---

### Tokenization

Tokenize the text into words, punctuation, or meaningful components.

**Example Code**:

```python
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]
```

---

### Stopword Removal

Remove common words that don't add much meaning to the text.

**Example Code**:

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]
```

---

### Stemming & Lemmatization

Convert words to their base or root form.

**Example Code**:

```python
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]
```

---

### Regex Preprocessing

Clean text further by using regular expressions.

**Example Code**:

```python
def regex_preprocessing(text):
    # Example: Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
```

---

### POS Tagging

Assign parts of speech to each word in the text.

**Example Code**:

```python
def pos_tagging(tokens):
    doc = nlp(" ".join(tokens))
    return [(token.text, token.pos_) for token in doc]
```

---

### Named Entity Recognition (NER)

Identify and classify named entities such as people, locations, and organizations.

**Example Code**:

```python
def ner(text):
    doc = nlp(text)
    return [(entity.text, entity.label_) for entity in doc.ents]
```

---

## Usage

After setting up the environment, you can use the following script to preprocess text:

```python
from preprocessing import clean_text, tokenize_text, remove_stopwords, apply_stemming, apply_lemmatization, pos_tagging, ner

# Example Text
text = "Hey! 👋 I just found this cool website 📱: https://www.example.com that has amazing articles on AI 🤖."

# Preprocessing Steps
cleaned_text = clean_text(text)
tokens = tokenize_text(cleaned_text)
tokens_no_stopwords = remove_stopwords(tokens)
tokens_stemmed = apply_stemming(tokens_no_stopwords)
tokens_lemmatized = apply_lemmatization(tokens_no_stopwords)

# POS Tagging
pos_tags = pos_tagging(tokens_lemmatized)

# Named Entity Recognition (NER)
entities = ner(text)

# Print results
print("Cleaned Text:", cleaned_text)
print("Tokens:", tokens)
print("Tokens after Stopword Removal:", tokens_no_stopwords)
print("Tokens after Stemming:", tokens_stemmed)
print("Tokens after Lemmatization:", tokens_lemmatized)
print("POS Tags:", pos_tags)
print("Named Entities:", entities)
```

---

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new pull request.
