import os
import re
import json
import time
import operator
from arm_tokenizer import tokenize
from nltk.tag.perceptron import PerceptronTagger
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import pandas as pd
from scipy.stats import wilcoxon

start_time = time.time()
PICKLE = "averaged_perceptron_tagger.pickle"

train_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/data/train_dataset/'
test_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/data/test_dataset/'
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
train_results_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/results/train_results/'
test_results_path = '/Users/hekpo/PycharmProjects/Karas/for_arm/results/test_results/'


def get_starts_of_paragraphs(paragraphs):
    start_of_paragraph = {}
    i = 1
    length = 0
    for item in paragraphs:
        start_of_paragraph[i] = length
        i += 1
        length += (len(item) + 2)
    return start_of_paragraph

def get_paragraphs_of(file_dir):
    with open(file_dir, encoding='utf-8')as fin:
        text = fin.read()
        white_line_regex = r"(?:\r?\n){2,}"
        paragraphs = re.split(white_line_regex, text.strip())
    return paragraphs

# punctuation tokenizer
punct_list = ['«', '»', '(', ')', '/', '\\',
              ',', '.', '․', ':', '։',
              '՝', '՟', '՚',
              '՜', '՛', '՞', '—']

def punct_tokenizer(text):
    punct_tokens = []
    for item in text:
        if item in punct_list:
            punct_tokens.append(item)
    return punct_tokens

# stopwords tokenizer
stopwords_list = open('stopwords_hy.txt', encoding='utf-8').read().splitlines()

def stopword_tokenizer(text):
    word_tokens = tokenize(text)
    stopword_tokens = []
    for item in word_tokens:
        if item in stopwords_list:
            stopword_tokens.append(item)
    return stopword_tokens

# POS tokenizer
def pos_tokenizer(text):
    word_tokens = tokenize(text)
    # using pretrained model to tag all tokens
    pretrained_tagger = PerceptronTagger(load=True)
    results = pretrained_tagger.tag(word_tokens)
    # collecting pos from resulting tuples
    pos_tokens = []
    for word_pos in results:
        pos_tokens.append(word_pos[1])
    return pos_tokens

def work_with(file_dir, train_or_test): #train_or_test is a string just to separate results
    paragraphs = get_paragraphs_of(file_dir)
    p = 1
    num_of_paragr = len(paragraphs)
    """ if (num_of_paragr < p):
            m = 10
            sents = sent_tokenize(text)
            sent_num = len(sents)
            i = 0
            paragraphs = []
            while i + m < sent_num:
                sents_str = ' '.join([str(elem) for elem in sents[i:i + m]])
                paragraphs.append(sents_str)
                i += m
            sents_str = ' '.join([str(elem) for elem in sents[i:sent_num]])
            paragraphs.append(sents_str)
            num_of_paragr = (len(paragraphs))"""

    start_of_paragraphs = get_starts_of_paragraphs(paragraphs)

    # 1.word tfidf
    word_vectorizer = TfidfVectorizer(tokenizer=tokenize)
    word_vectors = word_vectorizer.fit_transform(paragraphs)
    # 2.punctation tfidf
    punct_vectorizer = TfidfVectorizer(tokenizer=punct_tokenizer)
    punct_vectors = punct_vectorizer.fit_transform(paragraphs)
    # 3.stopwords tfidf
    stopword_vectorizer = TfidfVectorizer(tokenizer=stopword_tokenizer)
    stopword_vectors = stopword_vectorizer.fit_transform(paragraphs)
    # 4.3-grams tfidf
    three_gram_vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(3, 3))
    three_gram__vectors = three_gram_vectorizer.fit_transform(paragraphs)
    # POS tfidf
    pos_vectorizer = TfidfVectorizer(tokenizer=pos_tokenizer)
    pos_vectors = pos_vectorizer.fit_transform(paragraphs)

    # concatenating calculated tfidfs
    vectors = sp.hstack((word_vectors, punct_vectors, stopword_vectors, three_gram__vectors, pos_vectors), format='csr')
    denselist = (vectors.todense()).tolist()
    df = pd.DataFrame(denselist)
    # doing Wilcoxon sign-rank test
    pvalues = {}
    for i in range(num_of_paragr - 1):
        stat, pvalue = wilcoxon(df.loc[i], df.loc[i + 1], alternative='two-sided')
        pvalues[i + 1] = pvalue
    # sorting pvalues in increasing  order
    sorted_pvalues = sorted(pvalues.items(), key=operator.itemgetter(1))

    # defining % of suspicious parts
    p = 0.3
    S = int(p * num_of_paragr) + 1
    alpha = 0.05
    style_change_borders = []
    # making a decision whether is a border
    for tpl in (sorted_pvalues[:S]):
        if tpl[1] < alpha:
            style_change_borders.append(start_of_paragraphs[tpl[0] + 1])  # +1 cause the numeration of paragraphs starts at 1
    style_change_borders.sort()

    if train_or_test == 'train':
        output_path = train_results_path
    elif train_or_test == 'test':
        output_path = test_results_path

    are_changes_exist = True
    if len(style_change_borders) == 0:
        are_changes_exist = False

    filename, file_format = os.path.splitext(os.path.basename(file_dir))
    with open(output_path + ('%s.truth' % filename), 'w') as o_f:
        # json.dump({"changes":are_changes_exist,"positions": style_change_borders}, o_f, indent=4)
        json.dump({"positions": style_change_borders}, o_f, indent=4)


def method_for_files(files):
    if files == train_files:
        path = train_path
        train_or_test = 'train'
    else:
        path = test_path
        train_or_test = 'test'
    for file in files:
        if file.endswith(".txt"):
            work_with(path + file, train_or_test)


method_for_files(test_files)
print("--- %s seconds ---" % (time.time() - start_time))
