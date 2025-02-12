import threading
import time
import psutil
import random
import re
import string
import nltk
import json

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



# ============================================================== FUNCTION TO AUGMENTED THE DATA ==============================================================
def load(filename):	
	with open(filename) as data_file:
		data = json.load(data_file)	

	return data

mydict = load('datasets/dict.json')
def get_sinonim(word):
    if word in mydict:
        synonyms = mydict[word]['sinonim']
        if synonyms:
            return random.choice(synonyms)
    return ""

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if get_sinonim(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_sinonim(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # Only replace up to n words
            break

    sentence = ' '.join(new_words)
    return sentence

def random_insertion(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        add_word(words)
    return ' '.join(words)

def add_word(words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = words[random.randint(0, len(words)-1)]
        synonyms = get_sinonim(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[random.randint(0, len(synonyms)-1)]
    random_idx = random.randint(0, len(words)-1)
    words.insert(random_idx, random_synonym)

def random_deletion(sentence, p=0.5):
    words = sentence.split()
    if len(words) == 1:
        return ' '.join(words)
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        return words[random.randint(0, len(words)-1)]
    return ' '.join(new_words)

def random_swap(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        words = swap_word(words)
    return ' '.join(words)

def swap_word(words):
    random_idx_1 = random.randint(0, len(words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(words)-1)
        counter += 1
        if counter > 3:
            return words
    words[random_idx_1], words[random_idx_2] = words[random_idx_2], words[random_idx_1]
    return words

def augment_data(data, num_augmentations=1):
    augmented_data = []
    for i in range(len(data)):
        sentence = data.iloc[i]['tweet']
        for _ in range(num_augmentations):
            augmentation_choice = random.choice([synonym_replacement, random_insertion, random_deletion, random_swap])
            augmented_sentence = augmentation_choice(sentence)
            augmented_data.append({'tweet': augmented_sentence, 'hatespeech': data.iloc[i]['hatespeech']})
    return pd.DataFrame(augmented_data)

# ============================================================== END ==============================================================


data_lock = Lock()

def simplifiedClass(row):
    for col in row.index:
        if col != 'Tweet' and row[col] == 1:
            return 1
    return 0

# Duration in seconds
duration = 1800
duration_training = 600
# duration = 180
# duration_training = 20

# Global model, results dictionary, and thread list for validation threads
validation_events = []
max_validation_threads = 150  # Set a maximum number of validation threads to prevent overloading
total_thread = []
stopword_factory = StopWordRemoverFactory()
stopword = stopword_factory.create_stop_word_remover()
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def preprocess(str):
    try:
        text = str.lower()
        strWithoutNum =  re.sub(r"\d+", "", text)
        text = strWithoutNum.translate(strWithoutNum.maketrans("","",string.punctuation)).strip()
        text = stopword.remove(text)
        text = stemmer.stem(text)
        return text
    except Exception as e:
        print(str)
        print(e)
        exit()

# Thread
    
def validation(stop_event, valid_event, thread_num):
    global X_train, y_train, result, x, test_df
    while not stop_event.is_set():
        if valid_event.is_set():
            break

        print("VALIDATION ", thread_num)
        x += 1
        if len(test_df) < 200:
            with data_lock:
                data = test_df
                test_df = test_df.drop(data.index)
        else:
            with data_lock:
                data = test_df.sample(random.randint(100, 200))
                test_df = test_df.drop(data.index)

        if len(data) <= 0:
            ori_data = ori_test_df.sample(random.randint(100, 200))
            aug_data = augment_data(ori_data, num_augmentations=1)
            data = aug_data

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
            print(data['tweet'])
            print("#######################################################################")
            event_stop.set()
            exit()
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
    x = tf.keras.layers.BatchNormalization()(x)  # Added batch normalization
    x = tf.keras.layers.Dropout(0.2)(x)  # Slightly reduced dropout
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.15)(x)  # Adjusted dropout
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    feature_layer = tf.keras.layers.Dense(64, activation='relu')(x)  # Reduced dimensionality in feature layer
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
    global zz, textclassifier, run_classification, extractor
    while not stop_event.is_set():
        try :
            zz += 1
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
            exit()


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
        # print(f'CPU : {cpu} --- MEMORY : {memory} --- TOTAL THREAD : {len(validation_events)}')
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
    
    X_train = X_train.apply(preprocess)

    ori_test_df = pd.DataFrame(test_data)
    test_df = ori_test_df
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
    
    pd.DataFrame(result).to_csv('resultnnnb/performance.csv', index=False)
    pd.DataFrame(performance).to_csv('resultnnnb/resource.csv', index=False)
    pd.DataFrame(total_thread).to_csv('resultnnnb/total_thread.csv', index=False)
    print("Done!")
    print(result)
    print(performance)