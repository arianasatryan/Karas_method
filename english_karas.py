import os
import re
import operator
import json
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import pandas as pd
from scipy.stats import wilcoxon

train_path='/Users/hekpo/PycharmProjects/Karas/for_eng/data/pan17-style-breach-detection-training-dataset-2017-02-15/'
test_path='/Users/hekpo/PycharmProjects/Karas/for_eng/data/pan17-style-breach-detection-test-dataset-2017-02-15/'
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
train_results_path='/Users/hekpo/PycharmProjects/Karas/for_eng/results/train_results/'
test_results_path='/Users/hekpo/PycharmProjects/Karas/for_eng/results/test_results/'

def get_starts_of_paragraphs(paragraphs):
    start_of_paragraph = {}
    i = 1
    length = 0
    for item in paragraphs:
        start_of_paragraph[i] = length
        i += 1
        length += (len(item)+2)
    return start_of_paragraph

def get_starts_of_sentsegs(sentseg):
    start_of_sentseg = {}
    i = 1
    length = 0
    for item in sentseg:
        start_of_sentseg[i] = length
        i += 1
        length += (len(item)+1)
    return start_of_sentseg

def get_paragraphs_of(file_dir):
    with open(file_dir, encoding='utf-8')as fin:
        text = fin.read()
        white_line_regex = r"(?:\r?\n){2,}"
        paragraphs = re.split(white_line_regex, text.strip())
    return paragraphs

def get_sentsegs_of(file_dir,m):
    with open(file_dir, encoding='utf-8')as fin:
        text = fin.read()
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
    return paragraphs


#punctuation tokenizer 
punct_list = [ '.', '?', '!',',',';',':','-','--','[',']','{','}','(',')','"']
def punct_tokenizer(text):
    punct_tokens = []
    for item in text:
        if item in punct_list:
            punct_tokens.append(item)
    return punct_tokens

# stopwords tokenizer
stopWords = set(stopwords.words('english'))
def stopword_tokenizer(text):
    word_tokens = word_tokenize(text)
    stopword_tokens = []
    for item in word_tokens:
        if item in stopWords:
            stopword_tokens.append(item)
    return stopword_tokens

#POS tokenizer
def pos_tokenizer(text):
    word_tokens = word_tokenize(text)
    results=pos_tag(word_tokens)
    pos_tokens = []
    for word_pos in results:
        pos_tokens.append(word_pos[1])
    return pos_tokens

# train_or_test is a string just to separate results_directories('train'or 'test')
#how_to_split is a string that specify method of splitting('par' or 'sent')
def work_with(file_dir, train_or_test,how_to_split,alpha):
    filename, file_format = os.path.splitext(os.path.basename(file_dir))
    if(how_to_split=='par'):
        segments = get_paragraphs_of(file_dir)
        starts_of_segments = get_starts_of_paragraphs(segments)
    elif(how_to_split=='sent'):
        segments = get_sentsegs_of(file_dir,m=5)
        starts_of_segments =  get_starts_of_sentsegs(segments)
    #conditional combined splitting ,example:if num_of_paragraphs==1 split by m sentences
    if(len(segments)<1):
        segments = get_sentsegs_of(file_dir, m=5)
        starts_of_segments = get_starts_of_sentsegs(segments)

    # 1.word tfidf
    word_vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    word_vectors = word_vectorizer.fit_transform(segments)
    # 2.punctation tfidf
    punct_vectorizer = TfidfVectorizer(tokenizer=punct_tokenizer)
    punct_vectors = punct_vectorizer.fit_transform(segments)
    # 3.POS tfidf
    pos_vectorizer = TfidfVectorizer(tokenizer=pos_tokenizer)
    pos_vectors = pos_vectorizer.fit_transform(segments)
    # 4.stopwords tfidf
    stopword_vectorizer = TfidfVectorizer(tokenizer=stopword_tokenizer)
    stopword_vectors = stopword_vectorizer.fit_transform(segments)
    # 5.3-grams tfidf
    three_gram_vectorizer = TfidfVectorizer(ngram_range=(3, 3))
    three_gram_vectors = three_gram_vectorizer.fit_transform(segments)

    # used features tf-idfs concatenating
    #vectors = sp.hstack((word_vectors,punct_vectors, pos_vectors, stopword_vectors, three_gram_vectors), format='csr')
    vectors = sp.hstack((word_vectors, pos_vectors, stopword_vectors), format='csr')
    denselist = (vectors.todense()).tolist()
    df = pd.DataFrame(denselist)

    # doing Wilcoxon sign-rank test
    num_of_segments=len(segments)
    pvalues = {}
    for i in range(num_of_segments - 1):
        stat, pvalue = wilcoxon(df.iloc[i], df.iloc[i + 1],zero_method='pratt', alternative='two-sided')
        pvalues[i + 1] = pvalue
    # sorting pvalues in increasing  order
    sorted_pvalues = sorted(pvalues.items(), key=operator.itemgetter(1))

    # defining % of suspicious parts
    p = 0.3
    S = int(p * num_of_segments) + 1
    style_change_borders = []
    # making a decision whether is a border
    for tpl in (sorted_pvalues[:S]):
        if tpl[1] <= alpha:
            style_change_borders.append(starts_of_segments[tpl[0] + 1])  # +1 cause the numeration of paragraphs starts at 1
    style_change_borders.sort()

    # writing results
    if train_or_test=='train':
        output_path = train_results_path
    elif train_or_test=='test':
        output_path = test_results_path

    with open(output_path+('%s.truth'%filename),'w') as o_f:
        json.dump({"borders": style_change_borders}, o_f, indent=4)

def method_for_files(files,how_to_split,alpha):
    if files == train_files:
        path = train_path
        train_or_test = 'train'
    else:
        path = test_path
        train_or_test = 'test'
    for file in files:
        if file.endswith(".txt"):
            work_with(path + file, train_or_test,how_to_split,alpha)



method_for_files(train_files,'par',alpha=1)


