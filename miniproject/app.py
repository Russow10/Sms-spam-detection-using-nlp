



import pandas as pd
import string
import re
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')



df = pd.read_csv('spam.csv', encoding='latin-1')


df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)


df.rename(columns={'v1': 'class', 'v2': 'message'}, inplace=True)



df.drop_duplicates(inplace=True)




df['label'] = df['class'].replace({'ham': 0, 'spam': 1})


df = df.drop(['class'], axis = 1)



def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

df['message']= df['message'].apply(lambda x:remove_punctuation(x))


df['message'] = df['message'].astype(str)


def tokenization(text):
    tokens = re.split('W+',text)
    return tokens
df['message']= df['message'].apply(lambda x: tokenization(x))


stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

df['message']= df['message'].apply(lambda x:remove_stopwords(x))


wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
df['message']=df['message'].apply(lambda x:lemmatizer(x))


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_test_str = [' '.join(tokens) for tokens in X_test]
X_train_str = [' '.join(tokens) for tokens in X_train]
vectorizer = CountVectorizer()
X_train_cv = vectorizer.fit_transform(X_train_str)
X_test_cv = vectorizer.transform(X_test_str)

model =  MultinomialNB()
model.fit(X_train_cv, y_train)



def sms(text):

    labels = ['not spam', 'spam']

    x = vectorizer.transform(text).toarray()

    predict = model.predict(x)

    prediction = predict[0]


    string = str(prediction)


    a = int(string)


    result = str("This Message is Looking " + labels[a])
    return result



from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def sms_check():
    if request.method == 'POST':
        text = request.form['text']
        results = sms([text])
        return render_template('result.html', text=text, result=results)
    return render_template('index.html')



    
if __name__ == '__main__':
    app.run(debug=True)





