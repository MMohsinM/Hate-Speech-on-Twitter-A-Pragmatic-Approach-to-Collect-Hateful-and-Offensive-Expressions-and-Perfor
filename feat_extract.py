from nltk import FreqDist
from nltk.util import ngrams
from nltk.lm import NgramCounter
from preprocess import get_attr
from copy import deepcopy
from preprocess import Tweet, time_it


# These are parts of speech tags corresponding to the noun, verb or (adjective or adverb)
utb_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']

ptb_tags ={}
ptb_tags['NN'] = "NOUN"
ptb_tags['NNS'] = "NOUN"
ptb_tags['NNP'] = "NOUN"
ptb_tags['NNPS'] = "NOUN"
ptb_tags['VB'] = "VERB"
ptb_tags['VBD'] = "VERB"
ptb_tags['VBG'] = "VERB"
ptb_tags['VBN'] = "VERB"
ptb_tags['VBP'] = "VERB"
ptb_tags['VBZ'] = "VERB"
ptb_tags['JJ'] = 'ADJECTIVE'
ptb_tags['JJR'] = 'ADJECTIVE'
ptb_tags['JJS'] = 'ADJECTIVE'
ptb_tags['RB'] = 'ADVERB'
ptb_tags['RBR'] = 'ADVERB'
ptb_tags['RBS'] = 'ADVERB'
ptb_tags['CC'] = 'COORDCONJUNCTION'
ptb_tags['CD'] = 'CARDINAL'
ptb_tags['DT'] = 'DETERMINER'
ptb_tags['EX'] = 'EXISTTHERE'
ptb_tags['FW'] = 'FOREIGNWORD'
ptb_tags['IN'] = 'PREPOSITION'
ptb_tags['LS'] = 'LISTMAKER'
ptb_tags['MD'] = 'MODAL'
ptb_tags['PDT'] = 'PREDETERMINER'
ptb_tags['POS'] = 'POSSESSIVEEND'
ptb_tags['PRP'] = 'PRONOUN'
ptb_tags['PRP$'] = 'PRONOUN'
ptb_tags['RP'] = 'PARTICLE'
ptb_tags['SYM'] = 'SYMBOL'
ptb_tags['TO'] = 'TO'
ptb_tags['UH'] = 'INTERJECTION'
ptb_tags['WDT'] = 'WHDETERMINER'
ptb_tags['WP'] = 'WHDETERMINER'
ptb_tags['WP$'] = 'WHDETERMINER'
ptb_tags['WRB'] = 'WHDETERMINER'
 

def delete_keys(num,class_dict, ratio_threshold=1.4):
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
                if (class_dict[num][common]/class_dict[deno][common])<ratio_threshold:
                    delete_num.append(common)
    return set(delete_num)


def attr_append_deepcopy(objs,attr1,attr2,cast):
    all_attr1 = list(get_attr(objs,attr1))
    all_attr2 = list(get_attr(objs,attr2))
    all_attr1 = deepcopy(all_attr1)
    for i in range(len(all_attr1)):
        for j in range(len(all_attr1[i])):
            all_attr1[i][j].append(all_attr2[i])
            all_attr1[i][j] = cast(all_attr1[i][j])
    return all_attr1


def frequency(all_pos):
    unigrams = [ngrams(pos,1) for pos in all_pos] # makes unigrams from pos
    counter = NgramCounter(unigrams) # initializes the counter object
    freq = counter.unigrams # gets the dict with pos as keys and frequencies as values
    return freq


def min_occ(freq, threshold):
    thres_freq = {}  # thres_freq keeps those freqs which are greater or equal to 9
    for key in freq.keys():
        if freq[key] >= threshold: 
            thres_freq[key]= freq[key]
    return thres_freq


def class_list(thres_freq, tags):
    hate = {}
    offn = {}
    clean = {}
    for key in thres_freq:
        if key[1] in tags:
            if key[2] =='clean':
                clean[key[0]] = thres_freq[key]
            elif key[2] =='offn':
                offn[key[0]] = thres_freq[key]
            elif key[2] =='hate':
                hate[key[0]] = thres_freq[key]
    return {'hate':hate,'offn':offn,'clean':clean}


def pop_class_dict(class_dict):
    delete_dict = {}
    for key in class_dict:
        delete_dict[key] = delete_keys(key,class_dict)
    # delete_dict = {'hate':delete_keys('hate',class_dict),
    #                 'offn':delete_keys('offn',class_dict),
    #                 'clean': delete_keys('clean',class_dict)}
    for dict in delete_dict:
        for key in delete_dict[dict]:
            class_dict[dict].pop(key)


def features(class_dict):
    features = []
    for key in class_dict:
        features += class_dict[key].keys()
    return features


def unigram_feat(tweets):
    '''
    Gets all the unigrams, counts frequency and keeps > 9
    Calculate class ratios. This implements unigram feature
    extraction from the paper.
    '''
    all_pos = attr_append_deepcopy(tweets, 'pos', 'label', tuple)
    freq = frequency(all_pos)
    thres_freq = min_occ(freq, threshold=9)
    class_dict = class_list(thres_freq, utb_tags)
    pop_class_dict(class_dict)
    return features(class_dict)

@time_it
def simplified_tag(all_pos):
    new_tags = []
    for pos in all_pos:
        sent = []
        for list_ in pos:
            if list_[1] in utb_tags:
                #score = Tweet.senti.getSentiment(list_[0])[0]
                score = Tweet.word_score(Tweet.sid.polarity_scores(list_[0]))
                if score>=0:
                    sentiment = "Positive_"
                else:
                    sentiment = 'Negative_'
            else:
                sentiment=""
            if list_[1] in ptb_tags:
                sent.append(sentiment+ptb_tags[list_[1]])
            else:
                sent.append(sentiment+".")
        new_tags.append((tuple(sent), list_[-1]))
    return new_tags


def get_all_pattern(all_tags, L=5):
    all_patterns = []
    for tags in all_tags:
        tags_sent = tags[0]
        patterns = tuple((tuple(tags_sent[i:i+L]),tags[1]) for i in range(len(tags_sent)-L+1))
        all_patterns += patterns
    return all_patterns


def pattern_feat(tweets):
    tweets_gte_5 = [tweet for tweet in tweets if len(tweet.tokens)>=5]
    all_pos = attr_append_deepcopy(tweets_gte_5, 'pos', 'label', tuple)
    new_tags = simplified_tag(all_pos)
    all_patterns = get_all_pattern(new_tags, L=5)
    hate=[]
    offn=[]
    clean=[]
    class_list = {'hate':hate,'offn':offn,'clean':clean}
    for patterns in all_patterns:
        class_list[patterns[1]].append(patterns[0])
    freq = {}
    for key in class_list:
        freq[key] = FreqDist(class_list[key])
    thres_freq = {}
    for key in freq:
        thres_freq[key] = min_occ(freq[key], threshold=7)
    delete_dict = {}
    for key in thres_freq:
        delete_dict[key] = delete_keys(key, thres_freq)
    for dict in delete_dict:
        for key in delete_dict[dict]:
            thres_freq[dict].pop(key)
    return features(thres_freq)


# all_pos = get_attr(tweets,'pos')
# unigram_feat(tweets)
# class_=[]
# masla = []
# for pos in all_pos:
#     for tup in pos:
#         if len(tup)>3:
#             masla.append([all_pos.index(pos),pos.index(tup)])
