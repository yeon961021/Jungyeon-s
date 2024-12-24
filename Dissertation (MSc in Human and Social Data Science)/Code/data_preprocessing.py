#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:25:27 2024

@author: ijeong-yeon
"""

import pandas as pd
from collections import Counter
import numpy as np
import editdistance
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords


"""
* Stop Words
1. English stopwords
2. Ethiopian stopwords
"""

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

english_vocab = set(words.words())
stop_words = set(stopwords.words('english'))
stopworld_eth = set(['iyo', 'ka', 'ay', 'ah', 'la', 'ha', 'kale', 'wa', 'si', 'kala', 'wax', 'ama', 'marka', 'bara', 'de', 'wey', 'en', 'ya', 'ta',
                'ye', 'ante', 'gin', 'hulu', 'ga', 'aho', 'hula', 'hin', 'rabbi', 'kan', 'nu', 'kana', 'ani', 'fi', 'al', 'alula', 'di', 'dan', 'badam'
                ,'shi', 'ly', 'kea', 'bel', 'blo', 'sile', 'gena', 'baa', 'naga', 'badan', 'na', 'hala', 'ey', 'bur', 'dal', 'dib', 'ale', 'ilka', 'mise', 'bal'
                , 'manta', 'ahey', 'lama', 'rag', 'haya', 'ayu', 'nay', 'yang', 'waar', 'anba', 'aa', 'io', 'sidi', 'harka', 'dha', 'san'
                , 'gara', 'yar', 'kalo', 'jeer', 'marae', 'lo', 'dari', 'ba', 'mid', 'dhan', 'bay', 'aha', 'mar', 'amba', 'kula',
                'el', 'es', 'se', 'con', 'un', 'las', 'para', 'pais', 'hay',  'das', 'ist', 'ich', 'den', 'sie', 'che'])

topic_keywords = ['tigray', 'ethiopia', 'tplf', 'war' , 'abiy', 'amhara', 'tdf', 'government', 'eritrea', 'ahmed', 'military', 'forces', 'battle','peace', 'ceasefire']

english_vocab.update(topic_keywords)

"""
Functions development for data preprocessing
"""

def word_frequency(df):
    text_series = df.iloc[:, 0].astype('string')
    text_series.dropna(inplace=True)

    text_data = text_series

    tokenized_text = [text.split() for text in text_data]

    word_counts = Counter(word for sentence in tokenized_text for word in sentence)

    word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    num_words = len(word_to_idx)
    co_occurrence_matrix = np.zeros((num_words, num_words))

    window_size = 2

    for sentence in tokenized_text:
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(0, idx - window_size): min(len(sentence), idx + window_size)]:
                if neighbor != word:
                    co_occurrence_matrix[word_to_idx[word]][word_to_idx[neighbor]] += 1

    top_words = word_counts.most_common(100)

    for word, count in top_words:
        print(f"{word}: {count}")

    return top_words

def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)

def remove_eth_stopwords(sentence):
    stopworld_eth = set(['iyo', 'ka', 'ay', 'ah', 'la', 'ha', 'kale', 'wa', 'si', 'kala', 'wax', 'ama', 'marka', 'bara', 'de', 'wey', 'en', 'ya', 'ta',
                'ye', 'ante', 'gin', 'hulu', 'ga', 'aho', 'hula', 'hin', 'rabbi', 'kan', 'nu', 'kana', 'ani', 'fi', 'al', 'alula', 'di', 'dan', 'badam'
                ,'shi', 'ly', 'kea', 'bel', 'blo', 'sile', 'gena', 'baa', 'naga', 'badan', 'na', 'hala', 'ey', 'bur', 'dal', 'dib', 'ale', 'ilka', 'mise', 'bal'
                , 'manta', 'ahey', 'lama', 'rag', 'haya', 'ayu', 'nay', 'yang', 'waar', 'anba', 'aa', 'io', 'sidi', 'harka', 'dha', 'san'
                , 'gara', 'yar', 'kalo', 'jeer', 'marae', 'lo', 'dari', 'ba', 'mid', 'dhan', 'bay', 'aha', 'mar', 'amba', 'kula',
                'el', 'es', 'se', 'con', 'un', 'las', 'para', 'pais', 'hay',  'das', 'ist', 'ich', 'den', 'sie', 'che'])
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stopworld_eth]
    return ' '.join(filtered_sentence)

def calculate_similarity(text1, text2):
    if len(text1) == 0 or len(text2) == 0:
        return 0.0
    distance = editdistance.eval(text1, text2)
    similarity_score = 1 - distance / max(len(text1), len(text2))
    return similarity_score

def remove_duplicates2(series):
    similarity_threshold = 0.9
    unique_texts = []
    duplicates_indices = []

    for idx, text in enumerate(series):
        is_duplicate = False
        for unique_text in unique_texts:
            if calculate_similarity(text, unique_text) > similarity_threshold:
                is_duplicate = True
                duplicates_indices.append(idx)
                break
        
        if not is_duplicate:
            unique_texts.append(text)
    unique_series = series.drop(index=duplicates_indices)
    
    return unique_series

def remove_duplicates(sentences):
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        if sentence not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence)

    return pd.Series(unique_sentences)


"""
Data preprocessing pipeline
"""
def data_preprocessing(df, english_vocab, stop_words, stopworld_eth, topic_keywords):

    df = pd.concat([df['Comment'], df['Video_Description']], ignore_index=True)
    df.reset_index(drop = True, inplace = True)
    initial_count = len(df)
    print(f"Initial data points: {initial_count}")

    # 1. Missing-value Handling
    text_series = df.astype('string')
    text_series.dropna(inplace=True)
    na_removed_count = len(text_series)
    print(f"Data points after removing NaNs: {na_removed_count} (Removed: {initial_count - na_removed_count})")

    # 2. English Vocabulary Filtering
    cleaned_text_series = text_series.apply(lambda text: ' '.join(token for token in text.split() if token.lower() in english_vocab))
    vocab_filtered_count = len(cleaned_text_series)
    print(f"Data points after English vocabulary filtering: {vocab_filtered_count} (Removed: {na_removed_count - vocab_filtered_count})")

    # 3. Stop-word Handling
    cleaned_text_series2 = cleaned_text_series.apply(remove_stopwords)
    stopwords_removed_count = len(cleaned_text_series2)
    print(f"Data points after stop-word removal: {stopwords_removed_count} (Removed: {vocab_filtered_count - stopwords_removed_count})")

    # 4. Ethiopian Languages Elimination
    cleaned_text_series3 = cleaned_text_series2.apply(remove_eth_stopwords)
    ethiopian_filtered_count = len(cleaned_text_series3)
    print(f"Data points after Ethiopian languages elimination: {ethiopian_filtered_count} (Removed: {stopwords_removed_count - ethiopian_filtered_count})")

    # 5. Duplicates Handling
    cleaned_text_series3.reset_index(drop = True, inplace = True)
    cleaned_series = remove_duplicates(cleaned_text_series3)
    duplicates_removed_count = len(cleaned_series)
    print(f"Data points after removing duplicates: {duplicates_removed_count} (Removed: {ethiopian_filtered_count - duplicates_removed_count})")
    
    # 6. 4 words or less sentence Removal
    cleaned_text_series = cleaned_series[cleaned_series.apply(lambda x: len(x.split()) > 3)]
    min_word_filtered_count = len(cleaned_text_series)
    print(f"Data points after removing sentences with <= 3 words: {min_word_filtered_count} (Removed: {duplicates_removed_count - min_word_filtered_count})")

    # 7. Keywords Filtering
    a = cleaned_text_series.str.lower().copy()
    filtered_comments = a.str.contains('|'.join(topic_keywords))
    final_series = a[filtered_comments]
    final_count = len(final_series)
    print(f"Data points after filtering by topic keywords: {final_count} (Removed: {min_word_filtered_count - final_count})")
    print("")

    final_series = final_series.reset_index(drop=True)
    return final_series

"""
Conducting data preprocessing
1) Tokenizing
2) Filtered by English Words
3) Stop Words Handling
4) Missing Value and Duplicated Data Removal
5) Valid Data Detection
"""

ex1 = pd.read_csv('data/phase_1.csv')
ex2 = pd.read_csv('data/phase_2.csv')
ex3 = pd.read_csv('data/phase_3.csv')
ex4 = pd.read_csv('data/phase_4.csv')
ex5 = pd.read_csv('data/phase_5.csv')
ex6 = pd.read_csv('data/phase_6.csv')

def pipeline(ex1, ex2, ex3, ex4, ex5, ex6):
    print("Data Preprocessing - Phase 1")
    ex1 = data_preprocessing(ex1, english_vocab, stop_words, stopworld_eth, topic_keywords)
    print("Data Preprocessing - Phase 2")
    ex2 = data_preprocessing(ex2, english_vocab, stop_words, stopworld_eth, topic_keywords)
    print("Data Preprocessing - Phase 3")
    ex3 = data_preprocessing(ex3, english_vocab, stop_words, stopworld_eth, topic_keywords)
    print("Data Preprocessing - Phase 4")
    ex4 = data_preprocessing(ex4, english_vocab, stop_words, stopworld_eth, topic_keywords)
    print("Data Preprocessing - Phase 5")
    ex5 = data_preprocessing(ex5, english_vocab, stop_words, stopworld_eth, topic_keywords)
    print("Data Preprocessing - Phase 6")
    ex6 = data_preprocessing(ex6, english_vocab, stop_words, stopworld_eth, topic_keywords)
    
    ex_total = pd.concat([ex1, ex2, ex3, ex4, ex5, ex6])
    ex_total.reset_index(drop = True, inplace = True)
    ex_total.to_csv('data/new/cleaned_data.csv')    

    print("Data Preprocessing - Completed")
    
    return ex_total

df = pipeline(ex1, ex2, ex3, ex4, ex5, ex6)