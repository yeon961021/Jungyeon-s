#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:49:31 2024

@author: ijeong-yeon
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from IPython.display import clear_output
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import itertools
import random
from bertopic import BERTopic
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix as cm, ConfusionMatrixDisplay
import os
warnings.filterwarnings("ignore")

# Coherence score calculation function
def calculate_coherence_score(model, texts, topics):
    
    documents = pd.DataFrame({"Document": texts,
                          "ID": range(len(texts)),
                          "Topic": topics})

    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(cleaned_docs)
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names_out()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic)] 
               for topic in range(len(set(topics))-1)]

    # Evaluate (NPMI and Cv)
    coherence_model = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus,dictionary=dictionary, coherence='c_npmi')
    coherence = coherence_model.get_coherence()
    coherence_model2 = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus, dictionary=dictionary, coherence='c_v')
    coherence2 = coherence_model2.get_coherence()
    return coherence, coherence2

"""
Grid Search for hyperparameter tuning
"""
df = pd.read_csv('data/new/cleaned_data.csv', index_col=0)
df_grid = df.sample(n=5000, random_state=101)
texts = df_grid['0'].tolist()
full_texts = df['0'].tolist()

def create_model(params):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = UMAP(n_neighbors=params['umap_n_neighbors'], n_components=params['umap_n_components'], 
        metric='cosine', low_memory=True)
    hdbscan_model = HDBSCAN(min_cluster_size=params['hdbscan_min_cluster_size'], 
        metric='euclidean', prediction_data=True)
    model = BERTopic( umap_model=umap_model,hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,top_n_words=params['top_n_words'],language='english',
        calculate_probabilities=True,verbose=True )
    return model

# Define the parameter grid
param_grid = { 'umap_n_neighbors': [10, 15, 20],'umap_n_components': [5, 10, 15],
    'hdbscan_min_cluster_size': [5, 10, 15],'top_n_words': [10, 15, 20]}

# Identifying all possible combinations
models_per_day = 8
total_combinations = len(param_grid['umap_n_neighbors']) * len(param_grid['umap_n_components']) * len(param_grid['hdbscan_min_cluster_size']) * len(param_grid['top_n_words'])
combinations_per_day = total_combinations // models_per_day
all_combinations = list(itertools.product(param_grid['umap_n_neighbors'],
                                          param_grid['umap_n_components'],
                                          param_grid['hdbscan_min_cluster_size'],
                                          param_grid['top_n_words']))
random.shuffle(all_combinations)

for day in range(models_per_day):
    combinations_today = all_combinations[day * combinations_per_day:(day + 1) * combinations_per_day]
    print(f"Day {day + 1}:")
    for combination in combinations_today:
        umap_n_neighbors, umap_n_components, hdbscan_min_cluster_size, top_n_words = combination
        print(f"umap_n_neighbors={umap_n_neighbors}, umap_n_components={umap_n_components}, "
              f"hdbscan_min_cluster_size={hdbscan_min_cluster_size}, top_n_words={top_n_words}")
    print("\n")
    

def save_output(df, umap_n_neighbors, umap_n_components, hdbscan_min_cluster_size, top_n_words, model_name, coherence_score, coherence_score2):
    df_model = pd.DataFrame({
    'umap_n_neighbors': [umap_n_neighbors],
    'umap_n_components': [umap_n_components],
    'hdbscan_min_cluster_size': [hdbscan_min_cluster_size],
    'top_n_words': [top_n_words],
    'model_name': [model_name],
    'c_npmi_score' : [coherence_score], 'c_v_score' : [coherence_score2]})
    clear_output()
    result = pd.concat([df, df_model])
    return result

# Example 1
df_model = pd.read_csv("data/model_performance.csv", index_col=0)

umap_n_neighbors=20
umap_n_components=10
hdbscan_min_cluster_size=10
top_n_words=20
model_name = "81"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, low_memory=False)
hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, prediction_data= True)

model = BERTopic(umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    embedding_model=embedding_model,
    top_n_words=top_n_words,
    language='english',
    calculate_probabilities=True,
    verbose=True)

topics, _ = model.fit_transform(texts)
coherence_score, coherence_score2 = calculate_coherence_score(model, texts, topics)
df_model = save_output(df_model, umap_n_neighbors, umap_n_components, hdbscan_min_cluster_size, top_n_words, model_name, coherence_score, coherence_score2)
df_model.to_csv("data/model_performance.csv")

"""
Model Comparison (Quantitative + Qualitative)
1. Pick top 3 models based on Coherence scores (NPMI and cv)
2. Model visualisation
"""
# TOP 3 Models in both NPMI and Cv (Quant)
df_model = pd.read_csv("data/model_performance.csv", index_col=0)
top_3_npmi = df_model.nlargest(3, 'c_npmi_score')
top_3_cv = df_model.nlargest(3, 'c_v_score')
top_3_npmi
top_3_cv

# Model visualisation (Example: NPMI Model) (Qual)
umap_n_neighbors = 15
umap_n_components = 10
hdbscan_min_cluster_size = 15
top_n_words = 10

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, low_memory=False)
hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, prediction_data= True)

model = BERTopic(umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    embedding_model=embedding_model,
    top_n_words=top_n_words,
    language='english',
    calculate_probabilities=True,
    verbose=True)

topics, _ = model.fit_transform(texts)

model.visualize_topics() # Intertopic Distance Map

hierarchical_topics = model.hierarchical_topics(texts)
model.visualize_hierarchy(hierarchical_topics=hierarchical_topics) # Hierarchical Clustering


"""
Draft model construction (Cv Model)
"""
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

best_params = (10, 10, 5, 10)
df = pd.read_csv('data/new/cleaned_data.csv', index_col=0)
df = df['0'].tolist()

umap_n_neighbors, umap_n_components, hdbscan_min_cluster_size, top_n_words = best_params

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
umap_model = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, low_memory=False)
hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, prediction_data= True)

model = BERTopic(umap_model=umap_model,hdbscan_model=hdbscan_model,embedding_model=embedding_model,
    top_n_words=top_n_words,language='english',calculate_probabilities=True,verbose=True)

topics, _ = model.fit_transform(df)
directory = "model/bert_draft.json"
model.save(directory)

# Topic modeling output checking
topic_info = model.get_topic_info()
topic_info_sorted = topic_info.sort_values(by='Count', ascending=False)
topic_info_sorted

# Topic reducing with model visualisation (qualitative evaluation)
model.visualize_topics()
model = model.reduce_topics(df, nr_topics=380)
model.visualize_topics()

hierarchical_topics = model.hierarchical_topics(df)
model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

model = model.reduce_topics(df, nr_topics=190)
model.visualize_topics()

hierarchical_topics = model.hierarchical_topics(df)
model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

topic_info = model.get_topic_info()
topic_info_sorted = topic_info.sort_values(by='Count', ascending=False)
topic_info_sorted.to_csv('data/new/topic/overall.csv')
directory = "model/bert_final.json"
model.save(directory)

"""
Final model construction with 190 topics
"""
directory = "model/bert_final2.json"
model_final = BERTopic.load(directory)

phase_1 = pd.read_csv('data/new/phase_1.csv', index_col=0)
phase_1 = phase_1['0'].tolist()
phase_2 = pd.read_csv('data/new/phase_2.csv', index_col=0)
phase_2 = phase_2['0'].tolist()
phase_3 = pd.read_csv('data/new/phase_3.csv', index_col=0)
phase_3 = phase_3['0'].tolist()
phase_4 = pd.read_csv('data/new/phase_4.csv', index_col=0)
phase_4 = phase_4['0'].tolist()
phase_5 = pd.read_csv('data/new/phase_5.csv', index_col=0)
phase_5 = phase_5['0'].tolist()
phase_6 = pd.read_csv('data/new/phase_6.csv', index_col=0)
phase_6 = phase_6['0'].tolist()

df = pd.read_csv('data/new/cleaned_data.csv', index_col=0)
df_label = df.sample(n=4400, random_state=101)
df_label = df_label['0'].tolist()

topics, _ = model_final.transform(phase_1)
phase1 = pd.DataFrame({'discourse': phase_1,'topic': topics})
phase1.to_csv('data/new/topic/phase1.csv')

topics2, _ = model_final.transform(phase_2)
phase2 = pd.DataFrame({'discourse': phase_2,'topic': topics2})
phase2.to_csv('data/new/topic/phase2.csv')

topics3, _ = model_final.transform(phase_3)
phase3 = pd.DataFrame({'discourse': phase_3,'topic': topics3})
phase3.to_csv('data/new/topic/phase3.csv')

topics4, _ = model_final.transform(phase_4)
phase4 = pd.DataFrame({'discourse': phase_4,'topic': topics4})
phase4.to_csv('data/new/topic/phase4.csv')

topics5, _ = model_final.transform(phase_5)
phase5 = pd.DataFrame({'discourse': phase_5,'topic': topics5})
phase5.to_csv('data/new/topic/phase5.csv')

topics6, _ = model_final.transform(phase_6)
phase6 = pd.DataFrame({'discourse': phase_6,'topic': topics6})
phase6.to_csv('data/new/topic/phase6.csv')

topics_test1, _ = model_final.transform(df_label) # with 4,400 (10%)
test1 = pd.DataFrame({'discourse': df_label,'topic': topics_test1})
test1.to_csv('data/new/topic/test1.csv')

"""
Label evaluation
- This study assigned labels to each topics
- Therefore, it is required to check the performance
"""

topic_information = pd.read_csv('data/new/topic/overall.csv', index_col=0)
df = pd.read_csv('data/new/cleaned_data.csv', index_col=0)
df_label = df.sample(n=4400, random_state=101)
df_label = df_label['0'].tolist()
phase1 = pd.read_csv('data/new/topic/phase1.csv', index_col=0)
phase2 = pd.read_csv('data/new/topic/phase2.csv', index_col=0)
phase3 = pd.read_csv('data/new/topic/phase3.csv', index_col=0)
phase4 = pd.read_csv('data/new/topic/phase4.csv', index_col=0)
phase5 = pd.read_csv('data/new/topic/phase5.csv', index_col=0)
phase6 = pd.read_csv('data/new/topic/phase6.csv', index_col=0)

tigray_war_labels = {'Abiy Ahmed' : [4, 19, 32, 86, 126, 188],
'Anti-Abiy Ahmed' : [10, 22, 68, 96, 103, 132],
'Ethiopian Government' : 8,
'Ethiopian News and Media' : [0, 23, 38, 39, 41, 46, 49, 50, 60, 65, 69, 76, 79, 81, 89, 102, 105, 129, 141, 138, 159, 151, 149, 146, 145, 178, 177, 176],
'Humanitarian Crisis' : [21, 33, 51, 56, 73, 106, 156, 155, 144, 185, 184, 175, 174],
'International Politics' : [9, 26, 34, 53, 75, 128, 158, 157],
'Politics' : [137, 152, 153, 169],
'Religion' : [72, 101],
'Tigray War - Genocide' : 16,
'Tigray War - Anti-TPLF' : [11, 15, 24, 55, 59, 61, 62, 63, 67, 80, 97, 98, 112, 133, 139, 131, 143, 163, 162, 183, 187],
'Tigray War - Battle and Conflict' : [1, 14, 17, 18, 28, 42, 48, 57, 74, 84, 91, 90, 88, 100, 104, 118, 120, 121, 119, 130, 127, 140, 135, 136, 167],
'Tigray War - Ceasefire' : [6, 40, 43, 111, 113, 115, 165],
'Tigray War - Eritrea' : [3, 181],
'Tigray War - Peace' : [12, 44, 92, 94, 116, 123, 172, 170, 186, 180],
'Tigray War - Pro-TPLF' : [2, 5, 30, 64, 77, 78, 93, 107, 142, 171, 168, 161, 179],
'Tigray War - Propaganda' : [20, 37, 58],
'Tigray War - Tigray' : [25, 31, 36, 45, 70, 82, 109, 110, 114, 122, 125, 134, 147, 182],
'Tigray War - TPLF' : [7, 13, 27, 52, 71, 83, 85, 87, 108, 160, 150, 148, 164, 166],
'np.nan' : [-1, 29, 35, 47, 54, 66, 95, 99, 117, 124, 154, 173]}

def evaluation(y, test1, test2, test3, test4, test5):
    y = y['label']
    test1 = test1['label']
    test2 = test2['label']
    test3 = test3['label']
    test4 = test4['label']
    test5 = test5['label']
    acc1 = accuracy_score(y, test1)
    acc2 = accuracy_score(y, test2)
    acc3 = accuracy_score(y, test3)
    acc4 = accuracy_score(y, test4)
    acc5 = accuracy_score(y, test5) # Precision, Recall, F1

    avg_f1_score1 = f1_score(y, test1, average='macro')
    avg_f1_score2 = f1_score(y, test2, average='macro')
    avg_f1_score3 = f1_score(y, test3, average='macro')
    avg_f1_score4 = f1_score(y, test4, average='macro')
    avg_f1_score5 = f1_score(y, test5, average='macro')
    
    f1_scores1 = f1_score(y, test1, average=None)
    f1_scores2 = f1_score(y, test2, average=None)
    f1_scores3 = f1_score(y, test3, average=None)
    f1_scores4 = f1_score(y, test4, average=None)
    f1_scores5 = f1_score(y, test5, average=None)

    precision_scores1 = precision_score(y, test1, average='macro')
    precision_scores2 = precision_score(y, test2, average='macro')
    precision_scores3 = precision_score(y, test3, average='macro')
    precision_scores4 = precision_score(y, test4, average='macro')
    precision_scores5 = precision_score(y, test5, average='macro')

    recall_scores1 = recall_score(y, test1, average='macro')
    recall_scores2 = recall_score(y, test2, average='macro')
    recall_scores3 = recall_score(y, test3, average='macro')
    recall_scores4 = recall_score(y, test4, average='macro')
    recall_scores5 = recall_score(y, test5, average='macro')

    print(f"Test 1 Accuracy: {acc1:.4f}")
    print(f"Test 1 Average F1 Score: {avg_f1_score1:.4f}")
    print(f"Test 1 F1 Score: {f1_scores1}")
    print(f"Test 1 Precision: {precision_scores1:.4f}")
    print(f"Test 1 Recall: {recall_scores1:.4f}")
    
    print(f"Test 2 Accuracy: {acc2:.4f}")
    print(f"Test 2 Average F1 Score: {avg_f1_score2:.4f}")
    print(f"Test 2 F1 Score: {f1_scores2}")
    print(f"Test 2 Precision: {precision_scores2:.4f}")
    print(f"Test 2 Recall: {recall_scores2:.4f}")
    
    print(f"Test 3 Accuracy: {acc3:.4f}")
    print(f"Test 3 Average F1 Score: {avg_f1_score3:.4f}")
    print(f"Test 3 F1 Score: {f1_scores3}")
    print(f"Test 3 Precision: {precision_scores3:.4f}")
    print(f"Test 3 Recall: {recall_scores3:.4f}")
    
    print(f"Test 4 Accuracy: {acc4:.4f}")
    print(f"Test 4 Average F1 Score: {avg_f1_score4:.4f}")
    print(f"Test 4 F1 Score: {f1_scores4}")
    print(f"Test 4 Precision: {precision_scores4:.4f}")
    print(f"Test 4 Recall: {recall_scores4:.4f}")
    
    print(f"Test 5 Accuracy: {acc5:.4f}")
    print(f"Test 5 Average F1 Score: {avg_f1_score5:.4f}")
    print(f"Test 5 F1 Score: {f1_scores5}")
    print(f"Test 5 Precision: {precision_scores5:.4f}")
    print(f"Test 5 Recall: {recall_scores5:.4f}")

    acc = sum([acc1, acc2, acc3, acc4, acc5])/5
    avg_f1 = sum([avg_f1_score1, avg_f1_score2,avg_f1_score3,avg_f1_score4, avg_f1_score5])/5
    f1 = sum([f1_scores1, f1_scores2, f1_scores3, f1_scores4, f1_scores5])/5
    precision = sum([precision_scores1, precision_scores2, precision_scores3, precision_scores4, precision_scores5])/5
    recall = sum([recall_scores1, recall_scores2, recall_scores3, recall_scores4, recall_scores5])/5
    print(f'The overall accuracy is: {acc:.4f}')
    print(f'The overall average F1 score is: {avg_f1:.4f}')
    print(f'The overall F1 Score is: {f1}')
    print(f'The overall precision is: {precision:.4f}')
    print(f'The overall recall is: {recall:.4f}')

def get_label(topic):
    for label, topics in tigray_war_labels.items():
        if isinstance(topics, list) and topic in topics:
            return label
        elif topic == topics:
            return label
    return 'Unknown'

directory = "model/bert_final2.json"
model = BERTopic.load(directory)
test1['label'] = test1['topic'].apply(get_label)
topics_test2, _ = model.transform(df_label)
topics_test3, _ = model.transform(df_label)
topics_test4, _ = model.transform(df_label)                      
topics_test5, _ = model.transform(df_label)
topics_test6, _ = model.transform(df_label)

test2 = pd.DataFrame({'discourse': df_label,'topic': topics_test2})
test3 = pd.DataFrame({'discourse': df_label,'topic': topics_test3})
test4 = pd.DataFrame({'discourse': df_label,'topic': topics_test4})
test5 = pd.DataFrame({'discourse': df_label,'topic': topics_test5})
test6 = pd.DataFrame({'discourse': df_label,'topic': topics_test6})

test2['label'] = test2['topic'].apply(get_label)
test3['label'] = test3['topic'].apply(get_label)
test4['label'] = test4['topic'].apply(get_label)
test5['label'] = test5['topic'].apply(get_label)
test6['label'] = test6['topic'].apply(get_label)

evaluation(test1, test2, test3, test4, test5, test6) # Check the overall evaluation matrix

"""
Topic Analysis
"""
def label_trend(df, labels = tigray_war_labels):

    label_counts = {}

    for label, topics in labels.items():
        if not isinstance(topics, list):
            topics = [topics]
        total_count = df[df['Topic'].isin(topics)]['Count'].sum()
        label_counts[label] = total_count

    total_sum_excluding_nan = sum(count for label, count in label_counts.items() if label != 'np.nan')

    label_ratios = {label: count / total_sum_excluding_nan for label, count in label_counts.items() if label != 'np.nan'}

    df_label_counts = pd.DataFrame(list(label_counts.items()), columns=['Label', 'Count'])
    df_label_ratios = pd.DataFrame(list(label_ratios.items()), columns=['Label', 'Ratio'])

    df_label = df_label_counts.merge(df_label_ratios, on ='Label')
    df_label.sort_values(by='Ratio', ascending=False, inplace = True)

    peace1 = df_label[df_label['Label'] == 'Tigray War - Ceasefire']['Ratio'].values[0]
    peace2 = df_label[df_label['Label'] == 'Tigray War - Peace']['Ratio'].values[0]

    peace = round((peace1 + peace2), 5)
    nan_count = label_counts['np.nan']
    print(f'A total number of discourse from np.nan label: {nan_count}')
    print(f'Ratio of peace related topics : {(peace1 + peace2)*100:.2f}%')

    return df_label, peace

def trend_plot(df, title, color, output, ticks = 45):
    plot_labels = {'Abiy Ahmed' : 'Abiy Ahmed','Anti-Abiy Ahmed' : 'Anti\nAbiy Ahmed',
                   'Ethiopian Government' : 'Ethiopian\nGovernment',
                   'Ethiopian News and Media' : 'Ethiopian \nNews and Media',
                   'Humanitarian Crisis' : 'Humanitarian\nCrisis',
                   'International Politics' : 'International\nPolitics',
                   'Politics' : 'Politics','Religion' : 'Religion','Tigray War - Genocide' : 'Genocide',
                   'Tigray War - Anti-TPLF' : 'Anti-TPLF',
                    'Tigray War - Battle and Conflict' : 'Battle and\n Conflict',
                    'Tigray War - Ceasefire' : 'Ceasefire' ,'Tigray War - Eritrea' : 'Eritrea','Tigray War - Peace' : 'Peace',
                    'Tigray War - Pro-TPLF' : 'Pro-TPLF', 'Tigray War - Propaganda' : 'Propaganda',
                    'Tigray War - Tigray' : 'Tigray','Tigray War - TPLF' : 'TPLF','np.nan' : 'np.nan'}
    
    top_15 = df.sort_values(by='Count', ascending=False).head(15)
    top_15['Label'].replace(plot_labels, inplace=True)
    plt.figure(figsize=(10, 6))
    plt.plot(top_15['Label'], top_15['Count'], marker='o', linestyle='-', color=color)
    plt.xticks(rotation=ticks)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True)

    for label, count, ratio in zip(top_15['Label'], top_15['Count'], top_15['Ratio']):
        plt.text(label, count, f'{ratio*100:.2f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'topic/{output}.png', dpi=800, bbox_inches='tight')
    plt.show()

def topic_count_table(df):
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    return topic_counts

def label_visuals(df):
    df['Ratio'] = df['Ratio']
    word_freq = {word: df[df['Label'] == word]['Ratio'].sum() for word in df['Label'].unique()}
    
    wordcloud = WordCloud(
        width=1200,
        height=600,
        margin=1,
        background_color='White',
        max_words=150, max_font_size=100,
        min_font_size=15,
        random_state=22,
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Adding labels for all words regardless of their size
    for word, freq in word_freq.items():
        for item in wordcloud.layout_:
            if item[0] == word:
                x, y = item[1]
                plt.text(x, y + 10, f'({freq:.2f})',
                         fontsize=8, ha='center', va='center', color='black', fontweight='bold')

    plt.tight_layout(pad=0)
    plt.show()
    
# Example: Overall Trend
overall_trend, peace0 = label_trend(topic_information)
overall_trend
label_visuals(overall_trend)
trend_plot(overall_trend, 'Overall Topic Trend - Tigray War', 'green', output='overall_trend', ticks= 40)

# Example: Phase 1
phase1 = topic_count_table(phase1)
phase1_trend, peace1 = label_trend(phase1)
phase1_trend
label_visuals(phase1_trend)
trend_plot(phase1_trend, 'Phase 1 Topic Trend - Tigray War', 'navy', output='Phase 1', ticks= 40)

# Example: Trend of peace-related topics
peace_trend = {'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6'],
    'Ratio': [peace1, peace2, peace3, peace4, peace5, peace6]}

peace_df = pd.DataFrame(peace_trend)

plt.figure(figsize=(8, 5))
plt.plot(peace_df['Phase'], peace_df['Ratio'], marker='o', linestyle='-', color='g', label='Trend')

plt.title("Tigray War: Peace-Related Topics' Trend")
plt.xlabel(' ')
plt.ylabel('Topic Ratio')
plt.xticks(rotation=0)

for phase, ratio in zip(peace_df['Phase'], peace_df['Ratio']):
    plt.text(phase, ratio, f'{ratio*100:.2f}%', ha='center', va='bottom', fontsize=9, color='black')

plt.tight_layout()
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.savefig('topic/peace_trend.png', dpi=800, bbox_inches='tight')

plt.show()