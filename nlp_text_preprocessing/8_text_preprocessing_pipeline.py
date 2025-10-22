import nltk

#step1 tokenize
from nltk.tokenize import sent_tokenize, word_tokenize

#step 2 stopwords
from nltk.corpus import stopwords

#step 3 stemming and lemma

from nltk.stem import PorterStemmer, WordNetLemmatizer

#step 4 regex
import re

#pip install emoji 
# step 5
import emoji

# step 6
from nltk import pos_tag

#step 7
from nltk import ne_chunk

def download_packages(): 
    #downloading the required packages for processing
    nltk.download("punkt")
    nltk.download("punkt_tab")

    #stop words
    nltk.download("stopwords")
    # stemming and lemmetization
    nltk.download("wordnet")

    #POS and tagging 
    #POS
    # nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    # tagging
    nltk.download('tagsets_json')


    # named entity recognition package
    nltk.download('maxent_ne_chunker_tab')
    # nltk.download('maxent_ne_chunker')
    nltk.download('words')

# packages_names = []
# for package in packages_names:
#     nltk.download(package)

text = """"Hey! 👋 I just found this cool website 📱: https://www.example.com
 that has amazing articles on AI 🤖. I think @john_doe should check it out! It’s a great read, especially for tech enthusiasts like him. Have you seen the latest article by Dr. Alice Smith on the future of robotics? 🚀 I can't wait to learn more! 🔍 Also, don’t forget to follow their social media for updates 🌐."""
sent_tokens = sent_tokenize(text)
words_tokens = word_tokenize(text)

print("Sentence Tokens:")
print(sent_tokens)

print("\nWord Tokens:")
print(words_tokens)

stopping_words = set(stopwords.words("english"))
filtered_words = [word for word in words_tokens if word not in stopping_words]

print("\nFiltered Words (After Stopword Removal):")
print(filtered_words)

ps = PorterStemmer()
wnl = WordNetLemmatizer()
print("\nStemming Process:")
stemmed_words = []
for w in filtered_words:
    print(w, " --> ", ps.stem(w))
    stemmed_words.append(ps.stem(w))

print("Stemming words:\n", stemmed_words)

print("\nLemmatization Process:")
lemma_words = []
for w in filtered_words:
    print(w, " --> ", wnl.lemmatize(w))
    lemma_words.append(wnl.lemmatize(w))

print("Lemmatized words:\n", lemma_words)

# step 4 remove punctuations, extra spaces, lower case the sentences
listto_sentences = ' '.join(lemma_words)

lowercase_sentences = listto_sentences.lower()

print("\nLowercase Sentences:")
print(lowercase_sentences)

# Remove numbers and punctuation
clean_text = re.sub(r'[^a-zA-Z\s]', '', lowercase_sentences)

print("\nCleaned Text (No Punctuation, No Numbers):")
print(clean_text)

# Remove extra spaces
clean_text = re.sub(r'\s+', ' ', clean_text).strip()

print("\nCleaned Text (No Extra Spaces):")
print(clean_text)

# step 5 removing emoji, url, tags& mentions

clean_text = emoji.replace_emoji(text, replace='')

print("\nCleaned Text (After Removing Emojis, URLs, Mentions):")
print(clean_text)

# step 6
# tokenize the clean_text sentence 
text_tokens = word_tokenize(clean_text)

print("\nTokenized Cleaned Text:")
print(text_tokens)

# parts of speech tagging
pos = pos_tag(text_tokens)
print("\nPOS Tagging Results:")
print(pos)


# Step 7: Named Entity Recognition (NER)
# ner_words_token = ne_chunk(text_tokens)  # using tokenized words
ner_words = ne_chunk(pos)  # using POS tagged words
# print("\nNER Results (Using Tokenized Words):")
# print(ner_words_token)

print("\nNER Results (Using POS Tagged Words):")
print(ner_words)