import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv('flirt_dataset.csv')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# -------------------------
# Helper function to train pipeline
# -------------------------
def train_pipeline(model, vectorizer, model_name):
    pipe = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])

    pipe.fit(X_train, y_train)

    # Save pipeline (contains both model + vectorizer)
    with open(f"output/{model_name}_pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nModel: {model_name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return acc


# --------------------------------
# Train using TF-IDF
# --------------------------------
print("\n===== Training with TF-IDF =====")

results_tfidf = {
    "Logistic Regression": train_pipeline(LogisticRegression(max_iter=1000), TfidfVectorizer(), "lr_tfidf"),
    "SVC": train_pipeline(LinearSVC(), TfidfVectorizer(), "svc_tfidf"),
    "Random Forest": train_pipeline(RandomForestClassifier(n_estimators=100, random_state=42), TfidfVectorizer(), "rf_tfidf"),
    "Naive Bayes": train_pipeline(MultinomialNB(), TfidfVectorizer(), "nb_tfidf"),
}

# --------------------------------
# Train using CountVectorizer
# --------------------------------
print("\n===== Training with CountVectorizer =====")

results_cv = {
    "Logistic Regression": train_pipeline(LogisticRegression(max_iter=1000), CountVectorizer(), "lr_cv"),
    "SVC": train_pipeline(LinearSVC(), CountVectorizer(), "svc_cv"),
    "Random Forest": train_pipeline(RandomForestClassifier(n_estimators=100, random_state=42), CountVectorizer(), "rf_cv"),
    "Naive Bayes": train_pipeline(MultinomialNB(), CountVectorizer(), "nb_cv"),
}
# Display overall results
df_results = pd.DataFrame([results_tfidf, results_cv], index=['TF-IDF', 'CountVectorizer'])
print("\nOverall Results:\n", df_results)




# import pandas as pd
# import pickle
# # preprocessed data you can load and create model
# df = pd.read_csv('flirt_dataset.csv')
# print(df.head())
# #feature extraction and model training
# def text_representation_CV(X_train,X_test):
#     # Bag of Words
#     from sklearn.feature_extraction.text import CountVectorizer
#     vectorizer = CountVectorizer()
#     X_bow = vectorizer.fit_transform(X_train)
#     X_test_bow = vectorizer.transform(X_test)
#     print("Bag of Words representation shape:", X_bow.shape)
#     return X_bow,X_test_bow

# def text_representation_TFIDF(X_train,X_test):
#     # TF-IDF
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     tfidf_vectorizer = TfidfVectorizer()
#     X_train_tf = tfidf_vectorizer.fit_transform(X_train)
#     save_vectorizer(tfidf_vectorizer)
#     X_test_tf = tfidf_vectorizer.transform(X_test)
#     print("TF-IDF representation shape:", X_train_tf.shape)
#     return X_train_tf,X_test_tf


# def trainn_test_splitting(df):
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['text'], df['label'], test_size=0.2, random_state=42
#     )
#     return X_train, X_test, y_train, y_test


# #saving themodel
# count = 1
# def save_model(model):
#     # import pickle
#     global count
#     while count ==4:
#         break
#     with open(f'{"flirt_model_"+str(count)}.pkl', 'wb') as f:
#         pickle.dump(model, f)
#         count += 1
# def save_vectorizer(tfidf):
#     global count
#     while count ==4:
#         break
#     with open(f"{"vectorizer_"+str(count)}.pkl", "wb") as f:
#         pickle.dump(tfidf, f)
#         count += 1

        

# def logistic_regression_model(X_train_tfidf,X_test_tfidf ,y_train, y_test):
#     print("Logistic Regression model")
#     from sklearn.linear_model import LogisticRegression
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train_tfidf, y_train)
#     save_model(model)
#     accuracy = prediction_and_evaluation(model, X_test_tfidf, y_test)
#     return accuracy

# def svm_model(X_train_tfidf,X_test_tfidf ,y_train, y_test):
#     print("SVM model")
#     from sklearn.svm import LinearSVC
#     model = LinearSVC()
#     model.fit(X_train_tfidf, y_train)
#     save_model(model)
#     accuracy = prediction_and_evaluation(model, X_test_tfidf, y_test)
#     return accuracy

# def random_forest_model(X_train_tfidf,X_test_tfidf ,y_train, y_test):
#     print("Random Forest model")
#     from sklearn.ensemble import RandomForestClassifier

#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train_tfidf, y_train)
#     save_model(model)
#     accuracys = prediction_and_evaluation(model, X_test_tfidf, y_test)
#     return accuracys

# def naivebayes_model(X_train_tfidf,X_test_tfidf ,y_train, y_test):
#     print("Naive Bayes model")
#     from sklearn.naive_bayes import MultinomialNB
#     model = MultinomialNB()
#     model.fit(X_train_tfidf, y_train)
#     save_model(model)
#     accuracys = prediction_and_evaluation(model, X_test_tfidf, y_test)
#     return accuracys
# def prediction_and_evaluation(model, X_test, y_test):
#     from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
#     y_pred = model.predict(X_test)

#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Classification Report:\n", classification_report(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred)
#           )
#     return accuracy_score(y_test, y_pred)
    
# # text_representation_CV(df)
# X_train, X_test, y_train, y_test = trainn_test_splitting(df)
# X_train_tfidf,X_test_tfidf = text_representation_TFIDF(X_train,X_test)
# print("Training and Testing the model")
# df_results = pd.DataFrame(columns=['Logistic Regression', 'SVC', 'Random Forest', 'Naive Bayes'], index=['TF-IDF', 'CountVectorizer'])

# df_results.loc['TF-IDF','Logistic Regression'] = logistic_regression_model(X_train_tfidf,X_test_tfidf ,y_train, y_test)
# df_results.loc['TF-IDF', 'SVC'] = svm_model(X_train_tfidf,X_test_tfidf ,y_train, y_test)
# df_results.loc['TF-IDF','Random Forest'] = random_forest_model(X_train_tfidf,X_test_tfidf ,y_train, y_test)
# df_results.loc['TF-IDF', 'Naive Bayes'] = naivebayes_model(X_train_tfidf,X_test_tfidf ,y_train, y_test)
# print("\nOverall Results:\n", df_results)

# X_train, X_test, y_train, y_test = trainn_test_splitting(df)
# X_train_cv,X_test_cv = text_representation_CV(X_train,X_test)

# df_results.loc['CountVectorizer','Logistic Regression'] = logistic_regression_model(X_train_cv,X_test_cv ,y_train, y_test)
# df_results.loc['CountVectorizer', 'SVC'] = svm_model(X_train_cv,X_test_cv ,y_train, y_test)
# df_results.loc['CountVectorizer','Random Forest'] = random_forest_model(X_train_cv,X_test_cv ,y_train, y_test)
# df_results.loc['CountVectorizer', 'Naive Bayes'] = naivebayes_model(X_train_cv,X_test_cv ,y_train, y_test)
# print("\nOverall Results:\n", df_results)
