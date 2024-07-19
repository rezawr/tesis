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
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD


def simplifiedClass(row):
    return 1 if row['label'] == 1 else 0

newDf = {}
df = pd.read_csv('datasets/dataset3/preprocess_data.csv')
df['hatespeech'] = df.apply(simplifiedClass, axis=1)

newDf['tweet'] = df['text']
newDf['hatespeech'] = df['hatespeech']
newDf = pd.DataFrame(newDf)

testDf = pd.read_csv('datasets/dataset3/preprocess_valid_data.csv')
testDf['hatespeech'] = testDf.apply(simplifiedClass, axis=1)

newTestDf = {}
newTestDf['tweet'] = testDf['text']
newTestDf['hatespeech'] = testDf['hatespeech']
newTestDf = pd.DataFrame(newTestDf)

newDf = newDf.dropna(subset=['tweet'])
newDf._append(newTestDf)
print(newDf.head(5))

def mainLogisticRegression():
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(newDf['tweet']).toarray()
    y = newDf['hatespeech'].values

    svd = TruncatedSVD(n_components=512)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_oversampled, y_train_oversampled)
    y_pred = lr_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('done logistic regression')
    return accuracy, f1, recall, precision


def mainXGBoost():
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(newDf['tweet']).toarray()
    y = newDf['hatespeech'].values

    svd = TruncatedSVD(n_components=512)  # Adjust n_components based on your specific needs
    X_reduced = svd.fit_transform(X)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_oversampled, y_train_oversampled)
    y_pred = xgb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print('done xgboost')
    return accuracy, f1, recall, precision
    
if __name__ == "__main__":
    accLR, f1LR, recallLR, precissionLR = mainLogisticRegression()
    accXG, f1XG, recallXG, precissionXG = mainXGBoost()
    
    report_dict = {
        "Algorithm": ['Logistic Regression', 'XG Boost'],
        "Accuracy": [accLR, accXG],
        "Precision": [precissionLR, precissionXG],
        "Recall": [recallLR, recallXG],
        "F1-Score": [f1LR, f1XG],
    }
    
    pd.DataFrame(report_dict).to_csv('result_dataset3_plus.csv', index=False)