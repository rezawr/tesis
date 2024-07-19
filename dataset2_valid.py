import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD

def simplifiedClass(row):
    return 1 if row['sentimen'] not in [0, 1, 2] else 0

newDf = {}
df = pd.read_csv('datasets/dataset2/preprocess_data.csv')
df['hatespeech'] = df.apply(simplifiedClass, axis=1)

newDf['tweet'] = df['Tweet']
newDf['hatespeech'] = df['hatespeech']
newDf = pd.DataFrame(newDf)

newDf = newDf.dropna(subset=['tweet'])
print(newDf.head(5))
def mainNB():
    # 0.8223234624145785
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(newDf['tweet']).toarray()
    y = newDf['hatespeech'].values

    # Split the dataset
    svd = TruncatedSVD(n_components=512)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    
    nb_model = MultinomialNB()
    nb_model.fit(np.abs(X_train_oversampled), y_train_oversampled)
    y_pred = nb_model.predict(np.abs(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('done nb')
    return accuracy, f1, recall, precision

    
def mainAnnNB():
    # 0.8382687927107062
    # Feature extraction with TF-IDF
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(newDf['tweet']).toarray()
    y = newDf['hatespeech'].values
    
    # Split the dataset
    # Split the dataset
    svd = TruncatedSVD(n_components=512)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
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
    model.fit(X_train_oversampled, y_train_oversampled, epochs=5, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Define a new model for feature extraction
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
    X_train_features = feature_extractor.predict(X_train_oversampled)
    X_test_features = feature_extractor.predict(X_test)
    
    # The extracted features might not be suitable for MultinomialNB directly due to negative values.
    # Ensure to process or choose another model as necessary. This is a conceptual approach.
    
    # Train Naive Bayes on the extracted features
    nb_model = MultinomialNB()
    parameters = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],  # Add more values as needed
        'fit_prior': [True, False],
    }
    
    # Setup GridSearchCV
    clf = GridSearchCV(nb_model, parameters, cv=5, scoring='accuracy')
    clf.fit(np.abs(X_train_features), y_train_oversampled)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    # NB: You may need to normalize or adjust the features for MultinomialNB if they contain negative values.
    # nb_model.fit(np.abs(X_train_features), y_train)  # Using absolute values as a simple workaround
    y_pred = clf.predict(np.abs(X_test_features))
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('done ann nb')
    return accuracy, f1, recall, precision
    
def mainGridCV():
    #  0.8238420652999241
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(newDf['tweet']).toarray()
    y = newDf['hatespeech'].values

    # Split the dataset
    svd = TruncatedSVD(n_components=512)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    
    # Define the model and parameters for GridSearchCV
    nb_model = MultinomialNB()
    parameters = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],  # Add more values as needed
        'fit_prior': [True, False],
    }
    
    # Setup GridSearchCV
    clf = GridSearchCV(nb_model, parameters, cv=5, scoring='accuracy')
    clf.fit(np.abs(X_train_oversampled), y_train_oversampled)
    
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    
    # Applying the best model found by GridSearchCV on the test data
    y_pred = clf.predict(np.abs(X_test))
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('done nb fine tune')
    return accuracy, f1, recall, precision

def mainSVM():
    # 0.8375094912680334
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(newDf['tweet']).toarray()
    y = newDf['hatespeech'].values
    
    # Split the dataset
    svd = TruncatedSVD(n_components=512)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_oversampled, y_train_oversampled)
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('done svm')
    return accuracy, f1, recall, precision

def mainDT():
    #  0.7686661604657049
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(newDf['tweet']).toarray()
    y = newDf['hatespeech'].values

    # Split the dataset
    svd = TruncatedSVD(n_components=512)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)

    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, random_state=42)
    dt.fit(X_train_oversampled, y_train_oversampled)
    y_pred = dt.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('done dtree')
    return accuracy, f1, recall, precision
    
if __name__ == "__main__":
    accNB, f1NB, recallNB, precissionNB = mainNB()
    accGridCV, f1GridCV, recallGridCV, precissionGridCV = mainGridCV()
    accAnnNB, f1AnnNB, recallAnnNB, precissionAnnNB = mainAnnNB()
    accSVM, f1SVM, recallSVM, precissionSVM = mainSVM()
    accDT, f1DT, recallDT, precissionDT = mainDT()
    
    report_dict = {
        "Algorithm": ['ANN NB', 'NB', 'FTNB', 'SVM', 'DT'],
        "Accuracy": [accAnnNB, accNB, accGridCV, accSVM, accDT],
        "Precision": [precissionAnnNB, precissionNB, precissionGridCV, precissionSVM, precissionDT],
        "Recall": [recallAnnNB, recallNB, recallGridCV, recallSVM, recallDT],
        "F1-Score": [f1AnnNB, f1NB, f1GridCV, f1SVM, f1DT],
    }
    
    pd.DataFrame(report_dict).to_csv('result_dataset2.csv', index=False)