import pandas as pd

dataset=pd.read_csv("airline_sentiment_analysis.csv",encoding='latin1')
dataset.head()

for index, review in enumerate(dataset["text"][10:15]):
    print(index+1,".",review)


import re 
def clean (text): 
        
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()) 
  
dataset["text"] = [clean(i) for i in dataset["text"]]


def clean_text(text):
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    text = text.lower()
       
    return text


dataset['clean_text'] = dataset.text.apply(lambda x: clean_text(x))
dataset['clean_text']


STOP_WORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an', 'and',
              'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
              'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', "couldn't", 'did',
              "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'else', 'ever',
              'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
              'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',
              "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it',
              "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself',
              'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours',
              'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's",
              'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',
              'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
              "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",
              'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
              "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",
              'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']


def gen_freq(text):
    word_list = []

    for tw_words in text.split():
        word_list.extend(tw_words)

    word_freq = pd.Series(word_list).value_counts()
    
    word_freq = word_freq.drop(STOP_WORDS, errors='ignore')
    
    return word_freq


import matplotlib.pyplot as plt
from wordcloud import WordCloud

word_freq = gen_freq(dataset.clean_text.str)

wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


gen_freq(dataset.clean_text.str)[:30]


dataset.head()


import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector


tf_vector = get_feature_vector(np.array(dataset.iloc[:, 3]).ravel())
X = tf_vector.transform(np.array(dataset.iloc[:, 3]).ravel())
y = np.array(dataset.iloc[:, 1]).ravel()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)


y_predict_lr = LR_model.predict(X_test)
print(accuracy_score(y_test, y_predict_lr))


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict_lr)


def sentiment_prediction(text): 

   X_test=[text]
   print(X_test)
   test_df=pd.DataFrame()
   test_df["sentiment"]=["unknown"]
   test_df["text"]=X_test
   test_df["text"] = [clean(i) for i in test_df["text"]]
   spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]
   for char in spec_chars:
      test_df['text'] = test_df['text'].str.replace(char, ' ')

 
   test_df['text'] = test_df["text"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))


   test_feature = tf_vector.transform(np.array(test_df).ravel())

   # Using Logistic Regression model for prediction
   test_prediction_lr = LR_model.predict(test_feature)

   # Averaging out the hashtags result
   test_result_ds = pd.DataFrame({'hashtag': "#", 'prediction':test_prediction_lr})
   test_result = test_result_ds.groupby(['hashtag']).max().reset_index()
   test_result.columns = ['heashtag', 'predictions'] 
   res=test_result.predictions
   res = res.to_string(index = False)
   return str(res)






