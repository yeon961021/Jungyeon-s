#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:36:13 2024

@author: ijeong-yeon
"""

# Hugging Face Reset
from transformers import TRANSFORMERS_CACHE
import transformers
print(TRANSFORMERS_CACHE)

import shutil
shutil.rmtree(TRANSFORMERS_CACHE)
transformers.utils.move_cache()
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from pysentimiento import create_analyzer
from matplotlib.ticker import MaxNLocator

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix as cm, ConfusionMatrixDisplay

# Evaluation Matrix
def evaluation(y, pred):
    mapping = {'pos': 1, 'neu': 0, 'neg': -1}
    y = [mapping[val] for val in y]
    pred = [mapping[val] for val in pred]
    
    avg_f1_score = f1_score(y, pred, average='macro')
    f1_scores = f1_score(y, pred, average=None)
    precision_scores = precision_score(y, pred, average='macro')
    recall_scores = recall_score(y, pred, average='macro')
    acc = accuracy_score(y, pred)
    
    print('\n The overall accuracy is: ' + str(acc))
    print('\n The F1 scores for each of the classes are: ' + str(f1_scores))
    print('\n The average F1 score is: ' + str(avg_f1_score))
    print('\n The precision scores for each of the classes are: ' + str(precision_scores))
    print('\n The recall scores for each of the classes are: ' + str(recall_scores))
    
    conf_matrix = cm(y, pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['neg', 'neu', 'pos'])
    disp.plot()
    plt.show()

    return acc

def classify_sentiment(text, sm_model):
    scores = sm_model.polarity_scores(text)
    compound = scores['compound']
    
    if compound > 0.05:
        return 'pos'
    elif compound < -0.05:
        return 'neg'
    else:
        return 'neu'

models_score = []
df = pd.read_csv('data/sentiment_data.csv', index_col=0)
y = df['sentiment_mannual']
df = df.iloc[:, [0,1]] # positive points
df_sentiment = df

"""
Sentiment Analysis Model Comparison

Examples:
1. NLTK Sentiment Analysis model
2. Huggingface 1 - RoBERTa-base 1 (Reference: https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)_
"""

def nltk_model():
    nltk_sm = SentimentIntensityAnalyzer()
    res = {}
    for i in range(len(df_sentiment)):
        text = df_sentiment['text'][i]
        res[i] = nltk_sm.polarity_scores(text)
    df_nltk_sm = pd.DataFrame(res).T
    df_nltk_sm = df_nltk_sm.reset_index().rename(columns={'index': 'id'})
    df_nltk_sm = df_nltk_sm.merge(df_sentiment, how='left')
    df_nltk_sm['sentiment'] = df_nltk_sm['text'].apply(classify_sentiment, sm_model=nltk_sm)
    pred = df_nltk_sm['sentiment']
    acc1 = evaluation(y, pred)
    model1 = ("NLTK", acc1)
    models_score.append(model1)
    
def reberta_1():
    analyzer = create_analyzer(task="sentiment", lang="en")
    df_bert_1 = df_sentiment.copy()
    predictions = [analyzer.predict(text).output for text in df_bert_1['text']]
    df_bert_1['sentiment'] = [prediction.lower() for prediction in predictions]
    pred = df_bert_1['sentiment']
    acc4 = evaluation(y, pred)
    model4 = ("RoBERTa-base 1", acc4)
    models_score.append(model4)
    
# Model comparison
model_dic = {}

for model_name, accuracy in models_score:
    model_dic[model_name] = accuracy
    
df_model = pd.DataFrame(list(model_dic.items()), columns=['Model Name', 'Accuracy'])
df_model.sort_values(by='Accuracy', ascending=False)
df_model

"""
Constructing Sentiment Analysis Model
"""

def calculate_bounds(df):
    df['sentiment'] = df['sentiment'].replace({'neg': 'Negative', 'pos': 'Positive', 'neu': 'Neutral'})
    counts = df['sentiment'].value_counts()
    desired_order = ['Negative', 'Neutral', 'Positive']
    counts = counts.reindex(desired_order, fill_value=0)

    neg = counts['Negative']
    neu = counts['Neutral']
    pos = counts['Positive']
    
    neg_f1 = 0.78974359
    neu_f1 = 0.70387244
    pos_f1 = 0.51461988
    
    neg_conf = (0.7578, 0.8203)
    neu_conf = (0.6681, 0.7358)
    pos_conf = (0.4499, 0.5815)
    
    neg_based = round(neg/neg_f1)
    neu_based = round(neu/neu_f1)
    pos_based = round(pos/pos_f1)

    neg_lower = round(neg_based * neg_conf[0])
    neg_upper = round(neg_based * neg_conf[1])
    neu_lower = round(neu_based * neu_conf[0])
    neu_upper = round(neu_based * neu_conf[1])
    pos_lower = round(pos_based * pos_conf[0])
    pos_upper = round(pos_based * pos_conf[1])

    return df, (neg_lower, neg, neg_upper), (neu_lower, neu, neu_upper), (pos_lower, pos, pos_upper)

def sentiment_analysis(df):
    # Predict sentiments
    predictions = [analyzer.predict(text).get('output', 'neu') for text in df['0']]
    df['sentiment'] = [prediction.lower() for prediction in predictions]
    
    # Return the DataFrame with sentiments
    return df

def sentiment_visuals(df,title, neg_bound, neu_bound, pos_bound):
    # Define the ordered sentiments and their bounds
    ordered_sentiments = ['Negative', 'Neutral', 'Positive']
    bounds = {
        'Negative': neg_bound,
        'Neutral': neu_bound,
        'Positive': pos_bound
    }

    # Calculate sentiment counts and percentages
    sentiment_counts = df['sentiment'].value_counts().reindex(ordered_sentiments, fill_value=0)
    total_count = sentiment_counts.sum()
    percentages = (sentiment_counts / total_count) * 100

    # Extract counts and bounds
    sentiments = sentiment_counts.index
    counts = sentiment_counts.values
    upper_bound_values = [bounds[sentiment][2] for sentiment in ordered_sentiments]
    lower_bound_values = [bounds[sentiment][0] for sentiment in ordered_sentiments]

    upper_bound_percentages = [bounds[sentiment][2] / total_count * 100 for sentiment in ordered_sentiments]
    lower_bound_percentages = [bounds[sentiment][0] / total_count * 100 for sentiment in ordered_sentiments]

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Plot bars for sentiment counts
    plt.bar(sentiments, counts, color = ['#FF6347', '#FFA500', '#ADD8E6'], alpha=0.8, edgecolor='#E0FFFF')

    # Add lines for upper and lower bounds
    for sentiment, count, upper_bound, lower_bound in zip(sentiments, counts, upper_bound_values, lower_bound_values):
        plt.plot([sentiment, sentiment], [lower_bound, count], color='g', linestyle='--', label='Lower Bound' if sentiment == sentiments[0] else "", linewidth=2)
        plt.plot([sentiment, sentiment], [count, upper_bound], color='r', linestyle='--', label='Upper Bound' if sentiment == sentiments[0] else "", linewidth=2)

    # Annotate the plot with counts and percentages
    for sentiment, count, percentage, upper_bound, lower_bound, upper_bound_percentage, lower_bound_percentage in zip(
        sentiments, counts, percentages, upper_bound_values, lower_bound_values, upper_bound_percentages, lower_bound_percentages):
        plt.text(sentiment, count + 2, f'{int(count)} ({percentage:.1f}%)', ha='center', va='bottom', color='b', fontsize = 10)
        plt.text(sentiment, upper_bound + 2, f'{int(upper_bound)} ({upper_bound_percentage:.1f}%)', ha='center', va='bottom', color='r', fontsize = 10)
        plt.text(sentiment, lower_bound - 5, f'{int(lower_bound)} ({lower_bound_percentage:.1f}%)', ha='center', va='top', color='g', fontsize = 10)

    # Add labels, title, and legend
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title(title, fontsize = 14)
    plt.xticks(rotation=0, fontsize = 14)
    
    # Make sure that legends for upper and lower bounds are unique
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def sentiment_visuals2(df1, df2, title1, title2, neg_bound1, neu_bound1, pos_bound1, neg_bound2, neu_bound2, pos_bound2):
    # Create a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(28, 16))  # Increase the width to fit both plots
    colors = ['#FF6347', '#FFA500', '#ADD8E6']  # Define colors for bars
    sentiments = ['Negative', 'Neutral', 'Positive']  # Sentiments to be considered
    
    # Define bounds for each dataframe
    bounds_list = [
        {
            'Negative': neg_bound1,
            'Neutral': neu_bound1,
            'Positive': pos_bound1
        },
        {
            'Negative': neg_bound2,
            'Neutral': neu_bound2,
            'Positive': pos_bound2
        }
    ]
    
    # Iterate over the dataframes, titles, and axes
    for i, (df, title, bounds) in enumerate(zip([df1, df2], [title1, title2], bounds_list)):
        # Calculate sentiment counts and percentages
        sentiment_counts = df['sentiment'].value_counts().reindex(sentiments, fill_value=0)
        total_count = sentiment_counts.sum()
        percentages = (sentiment_counts / total_count) * 100

        # Extract counts and bounds
        counts = sentiment_counts.values
        upper_bound_values = [bounds[sentiment][2] for sentiment in sentiments]
        lower_bound_values = [bounds[sentiment][0] for sentiment in sentiments]

        upper_bound_percentages = [bounds[sentiment][2] / total_count * 100 for sentiment in sentiments]
        lower_bound_percentages = [bounds[sentiment][0] / total_count * 100 for sentiment in sentiments]

        # Plot bars for sentiment counts
        axes[i].bar(sentiments, counts, color=colors, alpha=0.8, edgecolor='#E0FFFF')

        # Add lines for upper and lower bounds
        for sentiment, count, upper_bound, lower_bound in zip(sentiments, counts, upper_bound_values, lower_bound_values):
            axes[i].plot([sentiment, sentiment], [lower_bound, count], color='g', linestyle='--', label='Lower Bound' if sentiment == sentiments[0] else "", linewidth=2)
            axes[i].plot([sentiment, sentiment], [count, upper_bound], color='r', linestyle='--', label='Upper Bound' if sentiment == sentiments[0] else "", linewidth=2)

        # Annotate the plot with counts and percentages
        for sentiment, count, percentage, upper_bound, lower_bound, upper_bound_percentage, lower_bound_percentage in zip(
            sentiments, counts, percentages, upper_bound_values, lower_bound_values, upper_bound_percentages, lower_bound_percentages):
            axes[i].text(sentiment, count + 2, f'{int(count)} ({percentage:.1f}%)', ha='center', va='bottom', color='b', fontsize=18)
            axes[i].text(sentiment, upper_bound + 2, f'{int(upper_bound)} ({upper_bound_percentage:.1f}%)', ha='center', va='bottom', color='r', fontsize=18)
            axes[i].text(sentiment, lower_bound - 5, f'{int(lower_bound)} ({lower_bound_percentage:.1f}%)', ha='center', va='top', color='g', fontsize=18)

        # Add labels, title, and legend
        axes[i].set_xlabel(' ')
        axes[i].set_ylabel(' ')
        axes[i].set_title(title, fontsize=24)
        axes[i].set_xticks(range(len(sentiments)))
        axes[i].set_xticklabels(sentiments, fontsize=20)
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i].set_yticklabels(axes[i].get_yticks(), fontsize=18)
        axes[i].legend(fontsize = 16)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def bounds(df):
    df['sentiment'] = df['sentiment'].replace({'neg': 'Negative', 'pos': 'Positive', 'neu': 'Neutral'})
    unique_classes = np.unique(df['sentiment'])
    class_counts = {cls: np.sum(np.array(df['sentiment']) == cls) for cls in unique_classes}
    df, neg_bound, neu_bound, pos_bound = calculate_bounds(df)
    return df, neg_bound, neu_bound, pos_bound

f1_scores = {'Negative': 0.789,'Neutral': 0.703,'Positive': 0.514}

analyzer = create_analyzer(task="sentiment", lang="en")

df = pd.read_csv('data/new/cleaned_data.csv', index_col=0)
phase_1 = pd.read_csv('data/new/phase_1.csv', index_col=0)
phase_2 = pd.read_csv('data/new/phase_2.csv', index_col=0)
phase_3 = pd.read_csv('data/new/phase_3.csv', index_col=0)
phase_4 = pd.read_csv('data/new/phase_4.csv', index_col=0)
phase_5 = pd.read_csv('data/new/phase_5.csv', index_col=0)
phase_6 = pd.read_csv('data/new/phase_6.csv', index_col=0)


# For example: Phase 1
phase_1 = sentiment_analysis(phase_1)
phase_1.to_csv('data/new/phase_1_sentiment.csv')
phase_1, neg_bound, neu_bound, pos_bound = bounds(phase_1)
title = 'Distribution of Sentiment in Social Media Discourse - Phase 1'
sentiment_visuals(phase_1, title ,neg_bound, neu_bound, pos_bound)

# For example: Phase 1 vs Phase 2
phase_2 = sentiment_analysis(phase_2)
phase_2.to_csv('data/new/phase_2_sentiment.csv')

phase_1, neg_bound1, neu_bound1, pos_bound1 = bounds(phase_1)
title1 = 'Distribution of Sentiment in Social Media Discourse - Phase 1'
phase_2, neg_bound2, neu_bound2, pos_bound2 = bounds(phase_2)
title2 = 'Distribution of Sentiment in Social Media Discourse - Phase 2'
sentiment_visuals2(phase_1, phase_2, title1, title2, neg_bound1, neu_bound1, pos_bound1, neg_bound2, neu_bound2, pos_bound2)