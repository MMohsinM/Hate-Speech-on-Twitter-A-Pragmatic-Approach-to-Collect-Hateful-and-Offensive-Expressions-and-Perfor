from nltk import FreqDist
from nltk.util import ngrams
from nltk.lm import NgramCounter
from preprocess import get_attr
import copy


# These are parts of speech tags corresponding to the noun, verb or (adjective or adverb)
ptb_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']


def delete_keys(num,class_dict):
    '''
    Implements the unigram ratio logic from
    the paper.
    #Args
        num :str key in class_dict
        class_dict: dictionary containing the nouns,verbs,(adjectives / adverbs)
    #Returns
        delete_num: set containing the keys to be deleted
    '''
    delete_num = []
    for deno in class_dict:
        if deno != num:
            commons = set(class_dict[num]).intersection(set(class_dict[deno]))
            for common in commons:
                if (class_dict[num][common]/class_dict[deno][common])<1.4:
                    delete_num.append(common)
    return set(delete_num)


def unigram_feat(tweets):
    '''
    Gets all the unigrams, counts frequency and keeps > 9
    Calculate class ratios. This implements unigram feature
    extraction from the paper.
    '''
    all_pos = get_attr(tweets,'pos')
    classes = list(get_attr(tweets,'label'))
    all_pos1 = copy.deepcopy(list(all_pos))
    for i in range(len(all_pos1)):
        for j in range(len(all_pos1[i])):
            all_pos1[i][j].append(classes[i])
            all_pos1[i][j] = tuple(all_pos1[i][j])
    unigrams = [ngrams(pos,1) for pos in all_pos1] # makes unigrams from pos
    counter = NgramCounter(unigrams) # initializes the counter object
    freq = counter.unigrams # gets the dict with pos as keys and frequencies as values
    thres_freq = {}  # thres_freq keeps those freqs which are greater or equal to 9
    for key in freq.keys():
        if freq[key] >= 9: 
            thres_freq[key]= freq[key]
    # Rewrite this part because class is actually hate, offensive and clean
    hate = {}
    offn = {}
    clean = {}
    for key in thres_freq:
        if key[1] in ptb_tags:
            if key[2] =='clean':
                clean[key[0]] = thres_freq[key]
            elif key[2] =='offn':
                offn[key[0]] = thres_freq[key]
            elif key[2] =='hate':
                hate[key[0]] = thres_freq[key]
    class_dict = {'hate':hate,'offn':offn,'clean':clean}
    delete_dict = {'hate':delete_keys('hate',class_dict),
                    'offn':delete_keys('offn',class_dict),
                    'clean': delete_keys('clean',class_dict)}
    for dict in delete_dict:
        for key in delete_dict[dict]:
            class_dict[dict].pop(key)
    unigram = []
    for key in class_dict:
        unigram += class_dict[key].keys()
    return unigram



def pattern_feat(tweets):
    '''
    Gets all the unigrams, counts frequency and keeps > 9
    Calculate class ratios. This implements unigram feature
    extraction from the paper.
    '''
    all_pos = get_attr(tweets,'pos')
    classes = list(get_attr(tweets,'label'))
    all_pos1 = copy.deepcopy(list(all_pos))
    for i in range(len(all_pos1)):
        for j in range(len(all_pos1[i])):
            all_pos1[i][j].append(classes[i])
            all_pos1[i][j] = tuple(all_pos1[i][j])
    unigrams = [ngrams(pos,1) for pos in all_pos1] # makes unigrams from pos
    counter = NgramCounter(unigrams) # initializes the counter object
    freq = counter.unigrams # gets the dict with pos as keys and frequencies as values
    thres_freq = {}  # thres_freq keeps those freqs which are greater or equal to 9
    for key in freq.keys():
        if freq[key] >= 9: 
            thres_freq[key]= freq[key]
    # Rewrite this part because class is actually hate, offensive and clean
    hate = {}
    offn = {}
    clean = {}
    for key in thres_freq:
        if key[1] in ptb_tags:
            if key[2] =='clean':
                clean[key[0]] = thres_freq[key]
            elif key[2] =='offn':
                offn[key[0]] = thres_freq[key]
            elif key[2] =='hate':
                hate[key[0]] = thres_freq[key]
    class_dict = {'hate':hate,'offn':offn,'clean':clean}
    delete_dict = {'hate':delete_keys('hate',class_dict),
                    'offn':delete_keys('offn',class_dict),
                    'clean': delete_keys('clean',class_dict)}
    for dict in delete_dict:
        for key in delete_dict[dict]:
            class_dict[dict].pop(key)
    unigram = []
    for key in class_dict:
        unigram += class_dict[key].keys()
    return unigram


# all_pos = get_attr(tweets,'pos')
# unigram_feat(tweets)