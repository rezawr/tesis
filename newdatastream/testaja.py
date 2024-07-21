import random
from nltk.corpus import wordnet
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json

factory = StemmerFactory()
stemmer = factory.create_stemmer()


def load(filename):	
	with open(filename) as data_file:
		data = json.load(data_file)	

	return data

mydict = load('datasets/dict.json')
def get_sinonim(word):
    if word in mydict:
        synonyms = mydict[word]['sinonim']
        if synonyms:
            ss = random.choice(synonyms)
            return ss
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
            if (isinstance(augmented_sentence, str) == False):
                print(sentence)
                print(augmented_sentence)
                print(augmentation_choice)
                exit()

            augmented_data.append({'tweet': augmented_sentence, 'hatespeech': data.iloc[i]['hatespeech']})
    return pd.DataFrame(augmented_data)

# Step 4: Scaling Up
# To scale this to generate 10 billion samples, follow a similar chunking and saving strategy as mentioned previously. Here’s how you can do it:

# python
# Copy code
# num_augmentations = 10  # Adjust this based on your needs and available memory
# total_samples_needed = 10000000000
# chunks = total_samples_needed // (len(original_data) * num_augmentations)

# for i in range(chunks):
#     augmented_data = augment_data(original_data, thesaurus, num_augmentations=num_augmentations)
#     augmented_data.to_csv(f'augmented_data_chunk_{i}.csv', index=False)  # Save to file to manage memory
#     print(f'Chunk {i+1}/{chunks} saved.')


def simplifiedClass(row):
    for col in row.index:
        if col != 'Tweet' and row[col] == 1:
            return 1
    return 0

newDf = {}
df = pd.read_csv('datasets/dataset1/preprocess_data.csv')
df['hatespeech'] = df.apply(simplifiedClass, axis=1)

newDf['tweet'] = df['Tweet']
newDf['hatespeech'] = df['hatespeech']
newDf = pd.DataFrame(newDf)

newDf = newDf.dropna(subset=['tweet'])

augmented_data = augment_data(newDf, num_augmentations=1)
augmented_data.to_csv('newdatastream/cobacoba.csv', index=False)
print(augmented_data)
