import pickle
import pandas as pd

# Load saved pipeline
pipe = pickle.load(open("output/lr_tfidf_pipeline.pkl", "rb"))

df = pd.read_csv("whatsapp_chat_data.csv")

preds = pipe.predict(df["Text"])

print(preds)
df['label'] = preds
# print(df[['Text', 'label']])
map_labels = {0: "Not Flirt", 1: "Flirt"} 
df['label'] = df['label'].map(map_labels)
print(df[['Text', 'label']])
df.to_csv("whatsapp_chat_predictions.csv", index=False)
    

