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
from threading import Lock
from sklearn.decomposition import TruncatedSVD


data_lock = Lock()

def simplifiedClass(row):
    for col in row.index:
        if col != 'Tweet' and row[col] == 1:
            return 1
    return 0

# Duration in seconds
duration = 3600
duration_training = 600
# duration = 180
# duration_training = 20

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

        data['tweet'] = data['tweet'].apply(preprocess)
        # Transforming new data using the fitted vectorizer
        data_vect = new_vectorize.transform(data['tweet'])
        data_reduced = new_svd.transform(data_vect)
        
        # Using the feature extractor to transform new data
        try :
            print(data_reduced.shape)
            features = extractor.predict(data_reduced)
            nb_pred = textclassifier.predict(np.abs(features))

            with data_lock:
                X_train = X_train._append(data['tweet'])
                y_train = y_train._append(pd.Series(nb_pred))
            
            result['accuracy'].append(accuracy_score(data['hatespeech'], nb_pred))
            result['f1'].append(f1_score(data['hatespeech'], nb_pred, average='weighted'))
            result['precision'].append(precision_score(data['hatespeech'], nb_pred, average='weighted'))
            result['recall'].append(recall_score(data['hatespeech'], nb_pred, average='weighted'))
            
            # print(result)
        except Exception as e :
            print("#######################################################################")
            print(e)
            print("#######################################################################")
            event_stop.set()
        finally:
            time.sleep(1)


def build_feature_extractor_model():
    global X_train, y_train, new_vectorize, extractor, new_svd
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

    svd = TruncatedSVD(n_components=512)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    # X = X.toarray()
    print("===>>> toarry done")
    # Split the dataset
    X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    print("===>>> split done")
    print("===>>> Shape of X after vectorization:", X.shape)
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_new_train, y_new_train)
    print("===>>> Shape of X_train_oversampled:", X_train_oversampled.shape)
    # ANN for feature extraction
    inputs = tf.keras.Input(shape=(X_train_oversampled.shape[1],))
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    feature_layer = tf.keras.layers.Dense(128, activation='relu')(x)  # This layer will be used for feature extraction
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(feature_layer)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile and train the ANN
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    early_stopping = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_oversampled, y_train_oversampled, epochs=5, batch_size=32, validation_data=(X_new_test, y_new_test), callbacks=[early_stopping])

    # Define a new model for feature extraction
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
    new_vectorize = vectorizer
    extractor = feature_extractor
    new_svd = svd
    print("===================================================================================")
    return feature_extractor, X_train_oversampled, y_train_oversampled

def train(stop_event):
    global y, textclassifier, run_classification, extractor
    while not stop_event.is_set():
        try :
            print("=================================================================================================")
            print("TRAINING ", y)
            print("=================================================================================================")
            y += 1
            extractor, x_train_oversampled, y_train_oversampled = build_feature_extractor_model()
            X_train_nn = extractor.predict(x_train_oversampled)
            tmp_classifier = GridSearchCV(MultinomialNB(), parameters, cv=5, scoring='accuracy')

            # Init First Train
            tmp_classifier.fit(X_train_nn, y_train_oversampled)
            textclassifier = tmp_classifier
            time.sleep(duration_training)
        except Exception as e:
            print("=================================================================================================")
            print(e)
            print("=================================================================================================")
            event_stop.set()


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
    y = 1
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
    exct, x_train_oversampled, y_train_oversampled = build_feature_extractor_model()
    X_train_nn = exct.predict(x_train_oversampled)
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

    parameters = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],  # Add more values as needed
        'fit_prior': [True, False],
    }
    textclassifier = GridSearchCV(MultinomialNB(), parameters, cv=5, scoring='accuracy')

    # Init First Train
    textclassifier.fit(X_train_nn, y_train_oversampled)
    print("ok")

    event_stop = threading.Event()
    run_classification = True
    valid_event = threading.Event()
    validation_events.append(valid_event)
    t1 = threading.Thread(target=validation, args=(event_stop,valid_event, 1)).start()
    t2 = threading.Thread(target=record_performance, args=(event_stop,)).start()

    time.sleep(duration_training)
    t3 = threading.Thread(target=train, args=(event_stop,)).start()

    time.sleep(duration)
    event_stop.set()
    
    # for x in validation_events:
    #     x.is_set()
    
    pd.DataFrame(result).to_csv('resultnnnb/performance.csv', index=False)
    pd.DataFrame(performance).to_csv('resultnnnb/resource.csv', index=False)
    pd.DataFrame(total_thread).to_csv('resultnnnb/total_thread.csv', index=False)
    print("Done!")
    print(result)
    print(performance)