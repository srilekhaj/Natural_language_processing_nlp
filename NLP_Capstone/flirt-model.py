import pandas as pd

# preprocessed data you can load and create model
df = pd.read_csv('flirt_dataset.csv')
print(df.head())
#feature extraction and model training
def text_representation_CV(X_train,X_test):
    # Bag of Words
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    print("Bag of Words representation shape:", X_bow.shape)
    return X_bow,X_test_bow

def text_representation_TFIDF(X_train,X_test):
    # TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tf = tfidf_vectorizer.transform(X_test)
    print("TF-IDF representation shape:", X_train_tf.shape)
    return X_train_tf,X_test_tf


def trainn_test_splitting(df):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def logistic_regression_model(X_train_tfidf,X_test_tfidf ,y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    accuracy = prediction_and_evaluation(model, X_test_tfidf, y_test)
    return accuracy

def svm_model(X_train_tfidf,X_test_tfidf ,y_train, y_test):
    from sklearn.svm import LinearSVC
    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)
    accuracy = prediction_and_evaluation(model, X_test_tfidf, y_test)
    return accuracy

def random_forest_model(X_train_tfidf,X_test_tfidf ,y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    accuracys = prediction_and_evaluation(model, X_test_tfidf, y_test)
    return accuracys

def naivebayes_model(X_train_tfidf,X_test_tfidf ,y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    accuracys = prediction_and_evaluation(model, X_test_tfidf, y_test)
    return accuracys
def prediction_and_evaluation(model, X_test, y_test):
    from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred)
          )
    return accuracy_score(y_test, y_pred)
    
# text_representation_CV(df)
X_train, X_test, y_train, y_test = trainn_test_splitting(df)
X_train_tfidf,X_test_tfidf = text_representation_TFIDF(X_train,X_test)
print("Training and Testing the model")
df_results = pd.DataFrame(columns=['Logistic Regression', 'SVC', 'Random Forest', 'Naive Bayes'], index=['TF-IDF', 'CountVectorizer'])

df_results.loc['TF-IDF','Logistic Regression'] = logistic_regression_model(X_train_tfidf,X_test_tfidf ,y_train, y_test)
df_results.loc['TF-IDF', 'SVC'] = svm_model(X_train_tfidf,X_test_tfidf ,y_train, y_test)
df_results.loc['TF-IDF','Random Forest'] = random_forest_model(X_train_tfidf,X_test_tfidf ,y_train, y_test)
df_results.loc['TF-IDF', 'Naive Bayes'] = naivebayes_model(X_train_tfidf,X_test_tfidf ,y_train, y_test)
print("\nOverall Results:\n", df_results)

X_train, X_test, y_train, y_test = trainn_test_splitting(df)
X_train_cv,X_test_cv = text_representation_CV(X_train,X_test)

df_results.loc['CountVectorizer','Logistic Regression'] = logistic_regression_model(X_train_cv,X_test_cv ,y_train, y_test)
df_results.loc['CountVectorizer', 'SVC'] = svm_model(X_train_cv,X_test_cv ,y_train, y_test)
df_results.loc['CountVectorizer','Random Forest'] = random_forest_model(X_train_cv,X_test_cv ,y_train, y_test)
df_results.loc['CountVectorizer', 'Naive Bayes'] = naivebayes_model(X_train_cv,X_test_cv ,y_train, y_test)
print("\nOverall Results:\n", df_results)