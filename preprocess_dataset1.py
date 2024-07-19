import nltk
# nltk.download()


import pandas as pd
import re
import string

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

df = pd.read_csv('datasets/dataset1/data.csv', encoding='latin-1')

print(df)

print(df.HS.value_counts())

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
    print("==========================================================================")
    print(final)
    print("==========================================================================")
    return final


def main():
    print(df.head(5))
    df['Tweet'] = df['Tweet'].apply(preprocess)
    print(df['Tweet'])
    print(df.head(5))
    df.to_csv('datasets/dataset1/preprocess_data.csv')
 
if __name__ == "__main__":
    main()