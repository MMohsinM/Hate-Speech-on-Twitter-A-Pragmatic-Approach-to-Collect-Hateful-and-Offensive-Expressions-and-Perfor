import re
import os
import subprocess as sbp
from nltk import download  as n_download
from nltk.stem import WordNetLemmatizer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
import pandas as pd
from split_hashtags import split_hashtag_to_words_all_possibilities as hd
from sentistrength import PySentiStr
import numpy as np
from tqdm import tqdm

from functools import wraps
import time
import os
def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result =  func(*args, **kwargs)
        t2 = time.time()
        print(f"{t2-t1} sec(s) for function {func.__name__}")  
        return result
    return wrapper

#df['column'] = df['column'].replace(old_val,new_val)

#from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.corpus import stopwords
#from nltk import download  as n_download
#from nltk import word_tokenize
#n_download('stopwords')
#n_download('averaged_perceptron_tagger')
# sample_tweet = "Hello @Mohsin checkout http://amxor.com"
# tweet = Tweet("My nigga how u doin?",'clean')
#data_folder = 'C:\zmyFIles\UET_SEMESTERS\semester7\ML\CEP\Hate Speech on Twitter A Pragmatic Approach\datasets'
#stop_words = stopwords.words("english")

data_files =  os.path.abspath("datasets\\hate_offensive_2.csv")

class Tweet:
    #Class varibles
    pos_input_path = 'tweet_token_input.txt'
    pos_output_path = 'tweet_pos_output.txt'
    contrast_words = ('but','however','in contrast','yet','differ','difference',
                    'variation', 'still','on the contrary','conversely','otherwise',
                    'although', 'nonetheless', 'despite', 'instead', 'alternatively',
                    'meanwhile', ',', '...', '?', '!', '.',';' 
                    )
                    #'on the other hand')
    negation_words = ('no', 'not', 'none', 'nobody', 'nothing', 'neither',
                     'hardly', 'barely', 'nowhere', 'never', "doesn't", "doesnt", 'aint',
                     'isnt', "isn't", 'wont', 'wasnt', "wasn't", "shan't", 'shouldnt',
                     "shouldn't", 'wouldnt', "wouldn't",  'cannot', "couldn't", 
                     "won't", "can't", "don't",'dont')
    slangs = pd.read_csv(".\\dictionary\\SlangSD.txt", sep='\t')
    slangs.columns = ['word','score']
    senti = PySentiStr()
    # Note: Provide absolute path instead of relative path
    senti.setSentiStrengthPath('C:\\java_modules\\twitie-tagger\\SentiStrength.jar') 
    senti.setSentiStrengthLanguageFolderPath('C:\\java_modules\\SentiStrength_Data')
    try:
        tk = TweetTokenizer()
    except LookupError:
        n_download('punkt')
        tk = TweetTokenizer()
    try:
        lemmatizer = WordNetLemmatizer() 
    except LookupError:
        n_download('wordnet')
        lemmatizer = WordNetLemmatizer() 
    try:
        sid = SentimentIntensityAnalyzer()
    except LookupError:
        n_download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()

    # static_method
    def word_score(scores):
        """
        #Args:
        scores : dict : SentimentIntensityAnalyser.polarity_scores
        #Returns
        score : int (-1 or 0 or 1)
        """
        if scores['compound'] >= 0.05:
            return 1
        elif scores['compound'] < 0.05 and scores['compound'] > -0.05:
            return 0
        else:
            return -1

    
    def __init__(self, text_string, label):
        self.raw = text_string
        self.label = label
        self.tidy = self.preprocess()
        self.tokens = self.basic_tokenize()
        self.pos = None #self.POS_gate_java() #This is a time consuming 
        # function so at initiallizing its set None
        # Manually set it by calling POS_gate_java or
        # set it in a batch
        self.lemma = self.lemmatization()
        self.negation_vector = self.make_negation_vector()


    def preprocess(self):
            """
            Accepts a text string and replaces:
            1) urls with nothing
            2) lots of whitespace with one instance
            3) mentions with nothing

            This allows us to get standardized counts of urls and mentions
            Without caring about specific people mentioned
            """
            space_pattern = r'[\s]+'
            giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            mention_regex = r'@[\w\-]+'
            hashtag_regex  = r'#[\w\-]+'
            end_space_regex = r'\s$'
            parsed_text = re.sub(giant_url_regex, '', self.raw)
            parsed_text = re.sub(mention_regex, '', parsed_text)
            parsed_text = re.sub(hashtag_regex, '', parsed_text)
            parsed_text = re.sub(space_pattern, ' ', parsed_text)
            parsed_text = re.sub(end_space_regex, '', parsed_text)
            self.tidy = parsed_text
            return parsed_text


    def basic_tokenize(self):
        """Same as tokenize but without the stemming"""
        self.tokens =  Tweet.tk.tokenize(self.tidy)
        #self.tokens = word_tokenize(self.tidy)
        return self.tokens


    def token_to_file(self):
        with open(self.pos_input_path, mode='w+') as f:
            f.write(" ".join(self.tokens))


    def POS_gate_java(self):
        self.token_to_file()
        x = sbp.check_output(["java", "-jar","twitie_tag.jar",
                            "models/gate-EN-twitter.model",
                            self.pos_input_path])
        x = x.decode().split('\n')[:-1]
        self.pos = [tuple(i.split("_")) for i in x[0].strip().split(" ")]
        return self.pos


    def lemmatization(self):
        """ 
        Does lemmatization on given list of tokens
        #Returns
        List of lemmas.
        """
        self.lemma = [Tweet.lemmatizer.lemmatize(token) for token in self.tokens]
        return self.lemma


    def make_negation_vector(self):
        """
        Recieves a list of tokens from a single tweet and generates
        the negation vector from it. A negation word (e.g., not and never)
        covers all the words that follow it until the next punctuation 
        mark or the occurrence of a contrast word (e.g., ``but,'' ``however,'' 
        etc). Words covered by a negation word are given a negation score
        equal to -1 while the rest of the words will be given a score
        equal to +1.
        """
        i = 0
        vector = []
        vector_entry = 1
        for token in self.tokens:
            if token in Tweet.negation_words:
                vector_entry = -1
            elif token in Tweet.contrast_words:
                vector_entry = 1
            vector.append(vector_entry)
        self.negation_vector = vector
        return vector
        #[Tweet.word_score(Tweet.sid.polarity_scores(token)) for token in self.tokens]


    def get_hashtags(self):
        '''
        Accepts a text string and returns the list of all hashtags 
        '''
        hashtag_regex = r'#[\w\-]+' #This is used to find hashtags
        hashtags = re.findall(hashtag_regex, self.raw)
        hashtags = list(map(lambda x: re.sub('#', '', x), hashtags)) # Removes # symbol from all hashtags found
        hashtags = [x.lower() for x in hashtags]
        return hashtags


    def hashtag_decomposition(self):
        '''
        Receives a list of hashtag of tweets and generate
        the corresponding sentence from it and return as
        a list
        '''
        hash_tokens = []
        for hashtag in self.get_hashtags():
            decomp = hd(hashtag)[0]
            if len(decomp)==0: #If no hashtag is returned
                hash_tokens.append(hashtag)
            else: #If 
                hash_tokens.append(decomp) 
        return hash_tokens


    def semantic_features(self):
        """
        the total score of positive words (PW),
        the total score of negative words (NW),
        the ratio of emotional (positive and negative)
            defined as: p(t) =  (PW-NW)/(PW-NW)
            p(t) is set to 0 if the tweet has no emotional words,
        the number of positive slang words,
        the number of negative slang words,
        the number of positive emoticons, <-- left this for now
        the number of negative emoticons, <-- left this for now 
        the number of positive hashtags, 
        the number of negative hashtags.
        """
        pos_slangs = 0 # number of positive slangs
        neg_slangs = 0 # number of negative slangs
        pos_hash = 0
        neg_hash = 0
        word_polar = [] # Contains the polarity of each word
        scores = []
        PW = 0
        NW = 0
        for word in self.tokens:
            score = Tweet.senti.getSentiment(word)[0]
            if score>0:
                if word in Tweet.slangs.word:
                    pos_slangs+=1
                PW += score
            elif score<0:
                if word in Tweet.slangs.word:
                    neg_slangs+=1
                NW += abs(score)
        if PW==0 and NW==0:
            ratio = 0 # p(t)
        else:
            ratio = (PW-NW)/(PW+NW) # p(t)
        hash_tokens = self.hashtag_decomposition()
        for tokens in hash_tokens: 
            if type(tokens) != str: # To convert the hashtags token to list 
                string = " ".join(tokens)
            else:
                string = tokens
            score = Tweet.senti.getSentiment(string)[0]
            if score>0:
                pos_hash +=1
            elif score<0:
                neg_hash += 1
        return (PW, NW, ratio, pos_slangs, neg_slangs, pos_hash, neg_hash)
                
    # Dont call it before setting the POS
    def syntatic_features(self):
        '''
        Returns:
            the number of exclamation marks,
            the number of question marks,
            the number of full stop marks,
            the number of all-capitalized words, 
            the number of quotes,
            the number of interjections,
            the number of laughing expressions, <-- Left this for now
            the number of words in the tweet.
        '''
        tidy = self.tidy
        word_count = len(tidy.split(" "))
        capital_count = len(re.findall(r"\b[A-Z]+\b", self.tidy))
        inter_count = 0
        if self.pos is None:
            self.POS_gate_java()
        for pos in self.pos:
            if pos == 'UH':
                inter_count += 1
        return (tidy.count("!"), tidy.count("?"), tidy.count("."), capital_count, 
                int(tidy.count("'")/2), inter_count, word_count)
    
#V_Tweet = np.vectorize(Tweet)

#encoding=encode)
def batch_to_file(tweets, encode='utf_8'):
    all_tokens = []
    for tweet in tweets:
        all_tokens.append(tweet.tokens)
    #df = pd.DataFrame(all_tokens)
    #df.to_csv(Tweet.pos_input_path,sep=' ', index=False)
    with open(tweets[0].pos_input_path, mode='w+' )as f:
        for i in tqdm(range(tweets.shape[0])):
            f.write(" ".join(tweets[i].tokens)+"\n")
        #for tweet in tweets:
        #    f.write(" ".join(tweet.tokens)+"\n")


def POS_gate_batch(path_in,path_out):
    '''
    #Args
    path_: file from which POS are written
            assumes that the required tokens
            are already present in it
    #Returns
    all_pos: a list of all POS tags in same 
            order
    '''
    x = sbp.check_output(["java", "-jar","twitie_tag.jar",
                        "models/gate-EN-twitter.model",
                        path_in])
    x = x.decode().split('\n')[:-1]
    all_pos = [list(i.strip().split(" ")) for i in x]
    # Loop converts the string sep by_ like man_NN
    # into tuple (man,NN) to change into standard NLTK output
    for sent in all_pos: # If time i will refactor it O(n^2)
        for i in range(len(sent)):
            sent[i] = tuple(sent[i].split("_")) 
    return all_pos
'''
def POS_gate_batch(path_in,path_out):   
    sbp.check_output(["java", "-jar","twitie_tag.jar",
                        "models/gate-EN-twitter.model",
                        path_in,'>', path_out])
    #x = x.decode().split('\n')[:-1]
    #Lets write to file first
    # Then we will read the file 
    x = pd.read_csv(path_out, sep=" ")
    all_pos = [list(i.strip().split(" ")) for i in x]
    # Loop converts the string sep by_ like man_NN
    # into tuple (man,NN) to change into standard NLTK output
    for sent in all_pos: # If time i will refactor it O(n^2)
        for i in range(len(sent)):
            sent[i] = tuple(sent[i].split("_")) 
    print(f"{t2-t1} sec(s) for List_comprehension")
    return all_pos
'''
# @time_it
# def init_tweet(df):
#     tweets = np.empty(len(df.index),dtype=object)
#     for index, row in df.iterrows():
#         tweets[index] = Tweet(row['tweet'], row['class'])
#     return tweets


# @time_it
# def init_tweet1(df):
#     VTweet = np.vectorize(Tweet)
#     return VTweet(df['tweet'], df['class'])
    

@time_it
def list_of_tweets(df):
    '''
    Returns a numpy array of tweets
    from a given DataFrame.
    '''
    VTweet = np.vectorize(Tweet)
    tweets= VTweet(df['tweet'], df['class'])
    batch_to_file(tweets)
    all_pos = POS_gate_batch(Tweet.pos_input_path,Tweet.pos_output_path)
    for i in range(len(all_pos)):
        tweets[i].pos = all_pos[i]
    return tweets
    
@time_it
def get_attr(arr, attr):
    ''' Creates a generator of attributes 
    from a given iter of objects
    #Args
        arr -- iterable containing objects
        attr -- (string) attribute name 
    #Returns
        generator of given attr in arr objects
    '''
    return (i.__getattribute__(attr) for i in arr)
        


@time_it
def unigram_features(tokens):
    unigram = None
    return unigram


# import numpy as np
# import pandas as pd
# from preprocess import list_of_tweets
# data = pd.read_csv('datasets\\data1.csv')
# x = list_of_tweets(data)
# np.save("tweet_obj.npy",x)