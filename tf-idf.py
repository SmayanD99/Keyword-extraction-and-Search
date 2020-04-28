#TF-IDF implementation including data preprocessing

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import operator
import statistics
from string import punctuation
stop_words = set(stopwords.words('english') + list(punctuation))
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def get_text_from_file(fname):
    """
    Get file from text doc
    """
    f=open(fname,'r')
    text=f.readlines()
    text=''.join(text) #converting the list to type str


def remove_string_special_characters(s):
    # Replace special character with ' '
    stripped = re.sub('[^\w\s]', '', s)
    stripped = re.sub('_', '', stripped)

    # Change any whitespace to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    
    return stripped

def count_words(text):
    """This function returns the 
    total number of words in the input text.
    """
    count = 0
    words = word_tokenize(text)
    for word in words:
        count += 1
    return count

def get_doc(text_sents_clean):
    """
    this function splits the text into sentences and
    considering each sentence as a document, calculates the 
    total word count of each.
    """
    doc_info = []
    i = 0
    for sent in text_sents_clean:
        i += 1 
        count = count_words(sent)
        temp = {'doc_id' : i, 'doc_length' : count}
        doc_info.append(temp)
    return doc_info

def create_freq_dict(sents):
    """
    This function creates a frequency dictionary
    of each document that contains words other than
    stop words.
    """
    i = 0
    freqDict_list = []
    for sent in sents:
        i += 1
        freq_dict = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word not in stop_words:
                if word in freq_dict:
                    freq_dict[word] += 1
                else:
                    freq_dict[word] = 1
                temp = {'doc_id' : i, 'freq_dict': freq_dict}
        freqDict_list.append(temp)
    return freqDict_list

def global_frequency(text_sents_clean):
    """
    This function returns a dictionary with the frequency 
    count of every word in the text
    """
    freq_table = {}
    text = ' '.join(text_sents_clean) #join the cleaned sentences to get the text 
    words = word_tokenize(text)
    for word in words:
        word = word.lower()
        word = ps.stem(word)
        if word not in stop_words:
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
    return freq_table

def get_keywords(text_sents_clean):
    """
    This function gets the top 5 most
    frequently occuring words in the whole text
    and stores them as keywords
    """
    freq_table = global_frequency(text_sents_clean)
    #sort in descending order
    freq_table_sorted = sorted(freq_table.items(), key = operator.itemgetter(1), reverse = True) 
    keywords = []
    for i in range(0, 5):  #taking first 5 most frequent words
        keywords.append(freq_table_sorted[i][0])
    return keywords

def computeTF(doc_info, freqDict_list):
    """
    tf = (frequency of the term in the doc/total number of terms in the doc)
    """
    TF_scores = []
    
    for tempDict in freqDict_list:
        id = tempDict['doc_id']
        for k in tempDict['freq_dict']:
            temp = {'doc_id' : id,
                    'TF_score' : tempDict['freq_dict'][k]/doc_info[id-1]['doc_length'],
                   'key' : k}
            TF_scores.append(temp)
    return TF_scores

def computeIDF(doc_info, freqDict_list):
    """
    idf = ln(total number of docs/number of docs with term in it)
    """
    
    IDF_scores = []
    counter = 0
    for dict in freqDict_list:
        counter += 1
        for k in dict['freq_dict'].keys():
            count = sum([k in tempDict['freq_dict'] for tempDict in freqDict_list])
            temp = {'doc_id' : counter, 'IDF_score' : math.log(len(doc_info)/count), 'key' : k}
    
            IDF_scores.append(temp)
                
    return IDF_scores


freqDict_list = create_freq_dict(text_sents_clean)
TF_scores = computeTF(doc_info, freqDict_list)
IDF_scores = computeIDF(doc_info, freqDict_list)

def computeTFIDF(TF_scores, IDF_scores):
    """
    TFIDF is computed by multiplying the coressponding
    TF and IDF values of each term. 
    """
    TFIDF_scores = []
    for j in IDF_scores:
        for i in TF_scores:
            if j['key'] == i['key'] and j['doc_id'] == i['doc_id']:
                temp = {'doc_id' : i['doc_id'],
                        'TFIDF_score' : j['IDF_score']*i['TF_score'],
                       'key' : i['key']}
        TFIDF_scores.append(temp)
    return TFIDF_scores

TFIDF_scores = computeTFIDF(TF_scores, IDF_scores)

def weigh_keywords(TFIDF_scores):
    """
    This function doubles the TFIDF score
    of the words that are keywords
    """
    keywords = get_keywords(text_sents_clean)
    for temp_dict in TFIDF_scores:
        if temp_dict['key'] in keywords:
            temp_dict['TFIDF_score'] *= 2
    return TFIDF_scores

TFIDF_scores = weigh_keywords(TFIDF_scores)



# This is the code for a single text file. 
# Another pyhton program needs to be written for multiple files.
# Also feeding the extracted keywords to the data structure yet to be done.
