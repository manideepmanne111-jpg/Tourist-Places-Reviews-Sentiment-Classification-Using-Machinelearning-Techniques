from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import webbrowser

main = Tk()
main.title("Tourist Place Reviews Sentiment Classification Using Machine Learning Techniques")
main.geometry("1300x1200")

sid = SentimentIntensityAnalyzer()

global filename
global X, Y
global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
global cv_X_train, cv_X_test, cv_y_train, cv_y_test
global tfidf_vectorizer
accuracy = []
precision = []
recall = []
fscore = []
city = []
place = []
global pos_count
global neg_count
global neu_cout

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []


def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():    
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.insert(END,filename+" loaded\n")
    

# def preprocess():
#     textdata.clear()
#     labels.clear()
#     city.clear()
#     place.clear()
#     text.delete('1.0', END)
#     dataset = pd.read_csv(filename,nrows=10000)
#     for i in range(len(dataset)):
#         msg = dataset.get_value(i, 'Raw_Review')
#         label = dataset.get_value(i, 'Rating')
#         cityname = dataset.get_value(i, 'City')
#         placename = dataset.get_value(i, 'Place')
#         msg = str(msg)
#         msg = msg.strip().lower()
#         if label >= 3:
#             labels.append(0)
#         else:
#             labels.append(1)
#         city.append(cityname)
#         place.append(placename)
#         clean = cleanPost(msg)
#         textdata.append(clean)
#         text.insert(END,clean+" ==== "+str(label)+"\n")

def preprocess():
    textdata.clear()
    labels.clear()
    city.clear()
    place.clear()
    text.delete('1.0', END)
    dataset = pd.read_csv(filename, nrows=10000)
    for i in range(len(dataset)):
        msg = dataset.at[i, 'Raw_Review']   # <-- changed
        label = dataset.at[i, 'Rating']     # <-- changed
        cityname = dataset.at[i, 'City']    # <-- changed
        placename = dataset.at[i, 'Place']  # <-- changed

        msg = str(msg).strip().lower()

        if label >= 3:
            labels.append(0)
        else:
            labels.append(1)

        city.append(cityname)
        place.append(placename)

        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END, clean + " ==== " + str(label) + "\n")


def TFIDFfeatureEng():
    text.delete('1.0', END)
    global Y
    global tfidf_vectorizer
    global tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=100)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:100]
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    tfidf_X_train, tfidf_X_test, tfidf_y_train, tfidf_y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal Reviews found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms using TFIDF : "+str(len(tfidf_X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms using TFIDF  : "+str(len(tfidf_X_test))+"\n")

def CVfeatureEng():
    text.delete('1.0', END)
    global cv_X_train, cv_X_test, cv_y_train, cv_y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    vectorizer = CountVectorizer()
    tfidf = vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:100]
    Y = np.asarray(labels)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    cv_X_train, cv_X_test, cv_y_train, cv_y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal Reviews found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total records used to train machine learning algorithms using Count Vector : "+str(len(cv_X_train))+"\n")
    text.insert(END,"Total records used to test machine learning algorithms using Count Vector  : "+str(len(cv_X_test))+"\n")    
        
def TFIDFClassification():
    text.delete('1.0', END)
    start = time.time()
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    cls = MultinomialNB()
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    p = precision_score(tfidf_y_test, predict,average='macro') * 100
    r = recall_score(tfidf_y_test, predict,average='macro') * 100
    f = f1_score(tfidf_y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Naive Bayes Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Naive Bayes Precision : '+str(p)+"\n")
    text.insert(END,'Naive Bayes Recall    : '+str(r)+"\n")
    text.insert(END,'Naive Bayes F1Score   : '+str(f)+"\n\n")

    cls = svm.SVC(class_weight='balanced')
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    p = precision_score(tfidf_y_test, predict,average='macro') * 100
    r = recall_score(tfidf_y_test, predict,average='macro') * 100
    f = f1_score(tfidf_y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'SVM Accuracy  : '+str(acc)+"\n")
    text.insert(END,'SVM Precision : '+str(p)+"\n")
    text.insert(END,'SVM Recall    : '+str(r)+"\n")
    text.insert(END,'SVM F1Score   : '+str(f)+"\n\n")

    cls = RandomForestClassifier()
    cls.fit(tfidf_X_train, tfidf_y_train)
    predict = cls.predict(tfidf_X_test)
    acc = accuracy_score(tfidf_y_test,predict)*100
    p = precision_score(tfidf_y_test, predict,average='macro') * 100
    r = recall_score(tfidf_y_test, predict,average='macro') * 100
    f = f1_score(tfidf_y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Random Forest Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Random Forest Precision : '+str(p)+"\n")
    text.insert(END,'Random Forest Recall    : '+str(r)+"\n")
    text.insert(END,'Random Forest F1Score   : '+str(f)+"\n")
    end = time.time()
    text.insert(END,"TFIDF Execution Time : "+str(end-start)+"\n\n")
    
def CVClassification():
    start = time.time()
    cls = MultinomialNB()
    cls.fit(cv_X_train, cv_y_train)
    predict = cls.predict(cv_X_test)
    acc = accuracy_score(cv_y_test,predict)*100
    p = precision_score(cv_y_test, predict,average='macro') * 100
    r = recall_score(cv_y_test, predict,average='macro') * 100
    f = f1_score(cv_y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Count Vector Naive Bayes Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Count Vector Naive Bayes Precision : '+str(p)+"\n")
    text.insert(END,'Count Vector Naive Bayes Recall    : '+str(r)+"\n")
    text.insert(END,'Count Vector Naive Bayes F1Score   : '+str(f)+"\n\n")

    cls = svm.SVC(class_weight='balanced')
    cls.fit(cv_X_train, cv_y_train)
    predict = cls.predict(cv_X_test)
    acc = accuracy_score(cv_y_test,predict)*100
    p = precision_score(cv_y_test, predict,average='macro') * 100
    r = recall_score(cv_y_test, predict,average='macro') * 100
    f = f1_score(cv_y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Count Vector SVM Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Count Vector SVM Precision : '+str(p)+"\n")
    text.insert(END,'Count Vector SVM Recall    : '+str(r)+"\n")
    text.insert(END,'Count Vector SVM F1Score   : '+str(f)+"\n\n")

    cls = RandomForestClassifier()
    cls.fit(cv_X_train, cv_y_train)
    predict = cls.predict(cv_X_test)
    acc = accuracy_score(cv_y_test,predict)*100
    p = precision_score(cv_y_test, predict,average='macro') * 100
    r = recall_score(cv_y_test, predict,average='macro') * 100
    f = f1_score(cv_y_test, predict,average='macro') * 100
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,'Count Vector Random Forest Accuracy  : '+str(acc)+"\n")
    text.insert(END,'Count Vector Random Forest Precision : '+str(p)+"\n")
    text.insert(END,'Count Vector Random Forest Recall    : '+str(r)+"\n")
    text.insert(END,'Count Vector Random Forest F1Score   : '+str(f)+"\n\n")
    end = time.time()
    text.insert(END,"Count Vectorizer Execution Time : "+str(end-start)+"\n\n")


def predict():
    text.delete('1.0', END)
    review = tf1.get()
    review = review.lower()
    sentiment_dict = sid.polarity_scores(review)
    negative = sentiment_dict['neg']
    positive = sentiment_dict['pos']
    neutral = sentiment_dict['neu']
    compound = sentiment_dict['compound']
    result = ''
    if compound >= 0.05 : 
        result = 'Positive' 
  
    elif compound <= - 0.05 : 
        result = 'Negative' 
  
    else : 
        result = 'Neutral'
    
    text.insert(END,review+' CLASSIFIED AS '+result+"\n\n")
    plt.pie([positive,negative,neutral],labels=["Positive","Negative","Neutral"],autopct='%1.1f%%')
    plt.title('Sentiment Graph')
    plt.axis('equal')
    plt.show()


def searchPlaces():
    global pos_count
    global neg_count
    global neu_cout
    dups = []
    text.delete('1.0', END)
    text.insert(END,"Similar Suggestion Places\n\n")
    review = simpledialog.askstring("Enter search place name here", "Please enter search place name here", parent=main)
    review = review.strip().lower()
    pos_count = 0
    neg_count = 0
    neu_count = 0
    suggestions = 0
    output = '<html><body>'
    output1 ='<table border="1" align="center"><tr><th>City Name</th><th>Place Name</th><th>Review</th><th>Sentiment Result</th></tr>'
    output2 ='<table border="1" align="center"><tr><th>City Name</th><th>Place Name</th><th>Review</th><th>Sentiment Result</th></tr>'
    output3 ='<table border="1" align="center"><tr><th>City Name</th><th>Place Name</th><th>Review</th><th>Sentiment Result</th></tr>'
    dataset = pd.read_csv(filename,nrows=500000,encoding='iso-8859-1')
    for i in range(len(dataset)):
        # msg = dataset.at(i, 'Review')
        # label = dataset.at(i, 'Rating')
        # cityname = dataset.at(i, 'City')
        # placename = dataset.at(i, 'Place')

        msg = dataset.at[i, 'Raw_Review']   # <-- changed
        label = dataset.at[i, 'Rating']     # <-- changed
        cityname = dataset.at[i, 'City']    # <-- changed
        placename = dataset.at[i, 'Place']  # <-- changed

        
        msg = str(msg)
        msg = msg.strip().lower()
        pname = placename.lower()
        if review in msg and placename not in dups:
            dups.append(placename)
            text.insert(END,"City : "+str(cityname)+" Place Name : "+str(placename)+"\n")
        if review in pname:
            #msg = textdata[i]
            sentiment_dict = sid.polarity_scores(msg)
            negative = sentiment_dict['neg']
            positive = sentiment_dict['pos']
            neutral = sentiment_dict['neu']
            compound = sentiment_dict['compound']
            result = ''
            #print(str(negative)+" "+str(neutral)+" "+str(positive))
            if compound >= 0.05 : 
                result = 'Positive'
                pos_count = pos_count + 1
                output1+='<tr><td>'+cityname+'</td><td>'+placename+'</td><td>'+msg+'</td><td>'+str(result)+'</td></tr>'
            elif compound <= - 0.05 : 
                result = 'Negative'
                neg_count = neg_count + 1
                output2+='<tr><td>'+cityname+'</td><td>'+placename+'</td><td>'+msg+'</td><td>'+str(result)+'</td></tr>'
            else :
                result = 'Neutral'
                neu_count = neu_count + 1
                output3+='<tr><td>'+cityname+'</td><td>'+placename+'</td><td>'+msg+'</td><td>'+str(result)+'</td></tr>'
            
    output+=output1+'</table><br/><br/>'+output2+'</table><br/><br/>'+output3+'</table><br/><br/>'
    output+='<table border=1 align=center>'
    output+='<tr><td>Positive Reviews Count</td><td>'+str(pos_count)+'</td></tr>'
    output+='<tr><td>Negative Reviews Count</td><td>'+str(neg_count)+'</td></tr>'
    output+='<tr><td>Neutral Reviews Count</td><td>'+str(neu_count)+'</td></tr>'
    if pos_count > neg_count and pos_count > neu_count:
        output+='<tr><td>Positive Reviews Maximum</td></tr>'
    elif neg_count > pos_count and neg_count > neu_count:
        output+='<tr><td>Negative Reviews Maximum</td></tr>'
    else:
        output+='<tr><td>Neutral Reviews Maximum</td></tr>'
    f = open("output.html", "w",encoding='iso-8859-1')
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)    
    
                                                         
                                                         
                
            




def graph():
    df = pd.DataFrame([['TFIDF Naive Bayes','Precision',precision[0]],['TFIDF Naive Bayes','Recall',recall[0]],['TFIDF Naive Bayes','F1 Score',fscore[0]],['TFIDF Naive Bayes','Accuracy',accuracy[0]],
                       ['Count Vector Naive Bayes','Precision',precision[3]],['Count Vector Naive Bayes','Recall',recall[3]],['Count Vector Naive Bayes','F1 Score',fscore[3]],['Count Vector Naive Bayes','Accuracy',accuracy[3]],
                       ['TFIDF SVM','Precision',precision[1]],['TFIDF SVM','Recall',recall[1]],['TFIDF SVM','F1 Score',fscore[1]],['TFIDF SVM','Accuracy',accuracy[1]],
                       ['Count Vector SVM','Precision',precision[4]],['Count Vector SVM','Recall',recall[4]],['Count Vector SVM','F1 Score',fscore[4]],['Count Vector SVM','Accuracy',accuracy[4]],
                       ['TFIDF Random Forest','Precision',precision[2]],['TFIDF Random Forest','Recall',recall[2]],['TFIDF Random Forest','F1 Score',fscore[2]],['TFIDF Random Forest','Accuracy',accuracy[2]],
                       ['Count Vector Random Forest','Precision',precision[5]],['Count Vector Random Forest','Recall',recall[5]],['Count Vector Random Forest','F1 Score',fscore[5]],['Count Vector Random Forest','Accuracy',accuracy[5]],
                      ],columns=['Metrics','Algorithms','Value'])
    df.pivot("Metrics", "Algorithms", "Value").plot(kind='bar')
    plt.show()
    
   
    
font = ('times', 15, 'bold')
title = Label(main, text='Tourist Place Reviews Sentiment Classification Using Machine Learning Techniques')
#title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Tourism Reviews Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

featureButton = Button(main, text="TFIDF Feature Extraction", command=TFIDFfeatureEng)
featureButton.place(x=20,y=200)
featureButton.config(font=ff)

traButton = Button(main, text="Count Vectorization Features Extraction", command=CVfeatureEng)
traButton.place(x=20,y=250)
traButton.config(font=ff)

clsButton = Button(main, text="Run SVM, Naive Bayes and Random Forest with TFIDF", command=TFIDFClassification)
clsButton.place(x=20,y=300)
clsButton.config(font=ff)

clsButton = Button(main, text="Run SVM, Naive Bayes and Random Forest with CountVector", command=CVClassification)
clsButton.place(x=20,y=350)
clsButton.config(font=ff)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=20,y=400)
graphButton.config(font=ff)

l1 = Label(main, text='Your Review :')
l1.config(font=font1)
l1.place(x=20,y=450)

tf1 = Entry(main,width=50)
tf1.config(font=font1)
tf1.place(x=150,y=450)

predictButton = Button(main, text="Predict Sentiments from Review", command=predict)
predictButton.place(x=20,y=500)
predictButton.config(font=ff)

predictButton = Button(main, text="Search Places", command=searchPlaces)
predictButton.place(x=20,y=550)
predictButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=550,y=100)
text.config(font=font1)

main.config()
main.mainloop()
