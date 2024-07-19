import threading
import time
import psutil
import random
import re
import string
import nltk

import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
# from sklearn.preprocessing import MinMaxScaler
from threading import Lock
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


data_lock = Lock()

def simplifiedClass(row):
    for col in row.index:
        if col != 'Tweet' and row[col] == 1:
            return 1
    return 0

# Duration in seconds
# duration = 3600
# duration_training = 600
duration = 180
duration_training = 20

# Global model, results dictionary, and thread list for validation threads
validation_events = []
max_validation_threads = 150  # Set a maximum number of validation threads to prevent overloading
total_thread = []

def lowercase(str):
    return str.lower()

def removeSpecialCharandExtraSpace(str):
    strWithoutNum =  re.sub(r"\d+", "", str)
    return strWithoutNum.translate(strWithoutNum.maketrans("","",string.punctuation)).strip()

def filtering(str):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()

    stop = stopword.remove(str)
    tokens = nltk.tokenize.word_tokenize(stop)
    return " ".join(tokens)

def stemming(str):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(str)

def preprocess(str):
    text = lowercase(str)
    text = removeSpecialCharandExtraSpace(text)
    text = filtering(text)
    final = stemming(text)
    return final

# Thread
def validation(stop_event, valid_event, thread_num):
    global X_train, y_train, result, x
    while not stop_event.is_set():
        if valid_event.is_set():
            print("=================================================================")
            print("VALIDATION ", thread_num, " IS STOPPED")
            print("=================================================================")
            break

        print("VALIDATION ", thread_num)
        x += 1
        data = test_df.sample(random.randint(100, 200))

        # data['tweet'] = data['tweet'].apply(preprocess)
        # Transforming new data using the fitted vectorizer
        data_vect = new_vectorize.transform(data['tweet'])
        data_reduced = new_svd.transform(data_vect)
        # data_scaled = new_scaler.transform(data_reduced)
        
        # Using the feature extractor to transform new data
        try :
            nb_pred = textclassifier.predict(data_reduced)

            with data_lock:
                X_train = X_train._append(data['tweet'])
                y_train = y_train._append(pd.Series(nb_pred))
                if len(X_train) != len(y_train):
                    print(f"After VALIDATION {thread_num} - X_train length: {len(X_train)}, y_train length: {len(y_train)}")
                    event_stop.set()
                # X_train = pd.concat([X_train, data['tweet']], ignore_index=True)
                # y_train = pd.concat([y_train, pd.Series(nb_pred)], ignore_index=True)

            with data_lock:
                result['accuracy'].append(accuracy_score(data['hatespeech'], nb_pred))
                result['f1'].append(f1_score(data['hatespeech'], nb_pred, average='weighted', zero_division=0))
                result['precision'].append(precision_score(data['hatespeech'], nb_pred, average='weighted', zero_division=0))
                result['recall'].append(recall_score(data['hatespeech'], nb_pred, average='weighted', zero_division=0))
            
            # print(result)
        except Exception as e :
            event_stop.set()
            time.sleep(10)
            print("#######################################################################")
            print(e)
            print("#######################################################################")
        finally:
            time.sleep(1)

def train(stop_event):
    global zz, textclassifier, run_classification, X_train, y_train, new_vectorize, new_svd, new_scaler
    while not stop_event.is_set():
        try :
            zz += 1
            with data_lock:
                priv_X_train = X_train
                priv_y_train = y_train
            # if len(priv_X_train) > 50000:
            #     priv_X_train = priv_X_train.tail(50000)
            #     priv_y_train = priv_y_train.tail(50000)

            print("===================================================================================")
            print("===>>> Length x:", len(priv_X_train))
            print("===>>> length y:", len(priv_y_train))
            vectorizer = TfidfVectorizer(max_features=50000)
            X = vectorizer.fit_transform(priv_X_train)
            y = priv_y_train.values
            print("===>>> tfidf done")

            svd = TruncatedSVD(n_components=100, random_state=42)  # Adjust n_components based on your specific needs
            X_reduced = svd.fit_transform(X)
            # X = X.toarray()
            print("===>>> toarry done")
            # scaler = MinMaxScaler()
            # X_scaled = scaler.fit_transform(X_reduced)
            # Split the dataset
            print("===>>> Shape of X after vectorization:", X.shape)
            smote = SMOTE(random_state=42)
            X_train_oversampled, y_train_oversampled = smote.fit_resample(X_reduced, y)
            print("===>>> Shape of X_train_oversampled:", X_train_oversampled.shape)
            tmp_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

            # Init First Train
            tmp_classifier.fit(X_train_oversampled, y_train_oversampled)
            textclassifier = tmp_classifier
            new_vectorize = vectorizer
            new_svd = svd
            # new_scaler = scaler
            time.sleep(duration_training)
        except Exception as e:
            event_stop.set()
            time.sleep(10)
            print("=================================================================================================")
            print(e)
            print("=================================================================================================")


def record_performance(stop_event):
    global performance, z
    threadnum = 2
    while not stop_event.is_set():
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        
        # Check if we should add a validation thread
        if cpu < 70 and memory < 70 and len(validation_events) < max_validation_threads:
            valid_event = threading.Event()
            validation_events.append(valid_event)
            threadnum+=1
            t = threading.Thread(target=validation, args=(stop_event,valid_event, threadnum))
            t.start()
            # print(f"Added a validation thread. Total: {len(validation_events)}")
        # Reduce to 1 thread if over threshold and more than 1 is running
        elif (cpu > 70 or memory > 70) and len(validation_events) > 1:
            while len(validation_events) > 1 and (cpu > 70 or memory > 70):
                thread_to_stop = validation_events.pop(1)
                thread_to_stop.set()
                # Not directly stopping the thread, but could signal it to stop if designed to listen for such a signal
                # print(f"Reduced validation threads. Total: {len(validation_events)}")

        total_thread.append(len(validation_events))
        performance['cpu'].append(cpu)
        performance['memory'].append(memory)
        print(f'CPU : {cpu} --- MEMORY : {memory} --- TOTAL THREAD : {len(validation_events)}')
        time.sleep(1)  # Check every 1 seconds


if __name__ == "__main__":
    x = 1
    zz = 1
    z = 1
    df = pd.read_csv('datasets/dataset1/data.csv', encoding='latin-1')
    df['hatespeech'] = df.apply(simplifiedClass, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['hatespeech'], random_state=0, train_size=0.7)
    test_data = {
        'tweet': X_test,
        'hatespeech': y_test
    }

    test_df = pd.DataFrame(test_data)
    # data = df.sample(random.randint(1, 100))
    # X_train.apply(preprocess)

    # Init training
    print("ok")

    print("===================================================================================")
    print("===>>> Length x:", len(X_train))
    print("===>>> length y:", len(y_train))
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(X_train)
    y = y_train.values
    print("===>>> tfidf done")

    svd = TruncatedSVD(n_components=100, random_state=42)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    # X = X.toarray()
    print("===>>> toarry done")
    # scaler = MinMaxScaler()
    # X_scaled = scaler.fit_transform(X_reduced)

    # Split the dataset
    print("===>>> Shape of X after vectorization:", X.shape)
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_reduced, y)
    print("===>>> Shape of X_train_oversampled:", X_train_oversampled.shape)
    
    parameters = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],  # Add more values as needed
        'fit_prior': [True, False],
    }
    tmp_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Init First Train
    tmp_classifier.fit(X_train_oversampled, y_train_oversampled)
    textclassifier = tmp_classifier
    new_vectorize = vectorizer
    new_svd = svd
    # new_scaler = scaler

    result = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }
    
    performance = {
        'cpu': [],
        'memory': [],
    }

    event_stop = threading.Event()
    run_classification = True
    valid_event = threading.Event()
    validation_events.append(valid_event)

    t3 = threading.Thread(target=train, args=(event_stop,)).start()

    t1 = threading.Thread(target=validation, args=(event_stop,valid_event, 1)).start()
    t2 = threading.Thread(target=record_performance, args=(event_stop,)).start()

    time.sleep(duration)
    event_stop.set()
    
    # for x in validation_events:
    #     x.is_set()

    time.sleep(10)
    print(len(result['accuracy']))
    print(len(result['f1']))
    print(len(result['precision']))
    print(len(result['recall']))
    pd.DataFrame(result).to_csv('resultxgb/performance.csv', index=False)
    pd.DataFrame(performance).to_csv('resultxgb/resource.csv', index=False)
    pd.DataFrame(total_thread).to_csv('resultxgb/total_thread.csv', index=False)
    print("Done!")
    print(result)
    print(performance)