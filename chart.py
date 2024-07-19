import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def simplifiedClass1(row):
    for col in row.index:
        if col != 'Tweet' and row[col] == 1:
            return 1
    return 0

def simplifiedClass2(row):
    return 1 if row['sentimen'] not in [0, 1, 2] else 0

def simplifiedClass3(row):
    return 1 if row['label'] == 1 else 0


df1 = pd.read_csv('datasets/dataset1/preprocess_data.csv')
df1['hatespeech'] = df1.apply(simplifiedClass1, axis=1)
df1['tweet'] = df1['Tweet']

df2 = pd.read_csv('datasets/dataset2/preprocess_data.csv')
df2['hatespeech'] = df2.apply(simplifiedClass2, axis=1)
df2['tweet'] = df2['Tweet']
df2 = df2.dropna(subset=['tweet'])
# newDf = newDf.dropna(subset=['tweet'])

df3 = pd.read_csv('datasets/dataset3/preprocess_data.csv')
df3['hatespeech'] = df3.apply(simplifiedClass3, axis=1)
testDf = pd.read_csv('datasets/dataset3/preprocess_valid_data.csv')
testDf['hatespeech'] = testDf.apply(simplifiedClass3, axis=1)
df3._append(testDf)
df3['tweet'] = df3['text']
df3 = df3.dropna(subset=['tweet'])

vectorizer1 = TfidfVectorizer(max_features=50000)
XTfidf1 = vectorizer1.fit_transform(df1['tweet']).toarray()
yTfidf1 = df1['hatespeech'].values

vectorizer2 = TfidfVectorizer(max_features=50000)
XTfidf2 = vectorizer2.fit_transform(df2['tweet']).toarray()
yTfidf2 = df2['hatespeech'].values

vectorizer3 = TfidfVectorizer(max_features=50000)
XTfidf3 = vectorizer3.fit_transform(df3['tweet']).toarray()
yTfidf3 = df3['hatespeech'].values



print("Dataset Dimension dataset 1:", XTfidf1.shape)
print("Dataset Dimension dataset 2:", XTfidf2.shape)
print("Dataset Dimension dataset 3:", XTfidf3.shape)




import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def outlier_iqr(data):
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    outliers = np.where((data < lower_bound) | (data > upper_bound), 1, 0)
    return np.mean(outliers, axis=0)

# z_scores = np.abs(stats.zscore(XTfidf1))
# outliers = (z_scores > 3).any(axis=1)
# outlier_ratio = np.mean(outliers)
# print("Outlier Ratio dataset 1:", outlier_ratio)


# z_scores = np.abs(stats.zscore(XTfidf2))
# outliers = (z_scores > 3).any(axis=1)
# outlier_ratio = np.mean(outliers)
# print("Outlier Ratio dataset 2:", outlier_ratio)

# z_scores = np.abs(stats.zscore(XTfidf3))
# outliers = (z_scores > 3).any(axis=1)
# outlier_ratio = np.mean(outliers)
# print("Outlier Ratio dataset 3:", outlier_ratio)


outlier_ratio_dataset1 = outlier_iqr(XTfidf1)
outlier_ratio_dataset2 = outlier_iqr(XTfidf2)

plt.figure(figsize=(10, 6))
plt.hist(outlier_ratio_dataset2, bins=50, alpha=0.75)
plt.title('Distribution of Outlier Ratios Across Features in Dataset 1')
plt.xlabel('Outlier Ratio')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print("Outlier Ratio dataset 1:", outlier_ratio_dataset1)
print("Outlier Ratio dataset 2:", outlier_ratio_dataset2)
print("Outlier Ratio dataset 3:", outlier_iqr(XTfidf3))