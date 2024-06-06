# models.py

from sentiment_data import List
from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np
import nltk
import random
import math

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        # raise Exception("Must be implemented")
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:       
        features = []
        for word in sentence:
            #Change to lower case, then get the index of the word 
            feature = self.indexer.add_and_get_index(word.lower(), add_to_indexer)
            features.append(feature)
        return Counter(features)

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        # raise Exception("Must be implemented")
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:      
        features = []
        
        #Combine word with its adjacent
        for i in range(len(sentence)-1):
            feature = self.indexer.add_and_get_index("|".join(sentence[i:i+2]), add_to_indexer)
            features.append(feature)
        
        return Counter(features)


#TF-IDF
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        # raise Exception("Must be implemented")
        self.indexer = indexer 
        self.word_freq = Counter() #Count world frequency in a document
        self.n_doc = 0 #Number of documents
        
    def get_indexer(self):
        return self.indexer
    
    #Calculate tf_idf
    def tf_idf(count, counter, n_doc, n_doc_with_word):
        tf = count / sum(counter.values())
        idf = math.log(n_doc / 1 + n_doc_with_word) #+1 for no 0, log for smoothing
        return tf * idf
    
    #Calculate word frequency and number of docs
    def count_word_freq(self, train_exs: List[SentimentExample]):
        self.n_doc = len(train_exs)
        for sentence in train_exs:
            words = list(sentence.words)
            for word in words:
                self.word_freq[word] += 1
                
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = []
        
        for word in sentence:
            feature = self.indexer.add_and_get_index(word, add_to_indexer)
            features.append(feature)
            
        counter = {}
        feature_counter = Counter(features)
        
        #If add_to_indexer == True, then return without further processing
        if add_to_indexer:
            return feature_counter
        
        #Get the tf_idf value
        for k, v in feature_counter.items():
            counter[k] = BetterFeatureExtractor.tf_idf(v, feature_counter, self.n_doc, self.word_freq[self.indexer.get_object(k)])
        
        return counter

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    def __init__(self, keys, featurizer):
        self.weights = np.zeros(max(keys)+1)
        self.featurizer = featurizer
    
    def predict(self, sentence: List[str]) -> int:
        
        fi = np.zeros(self.weights.shape)
        features = self.featurizer.extract_features(sentence, False)

        for k, v in features.items():
            fi[k] = v

        #Score = W*feature_vector
        score = np.dot(self.weights, fi)
        return 1 if score > 0.5 else 0
    
    def update(self, lr, sentence, label):
        fi = np.zeros(self.weights.shape)
        features = self.featurizer.extract_features(sentence)
        for k, v in features.items():
            fi[k] = v
            
        #Base on true label, update weights
        if label == 0:
            self.weights -= lr * fi
        else:
            self.weights += lr * fi


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, keys, featurizer):
        # raise Exception("Must be implemented")
        self.weights = np.random.random(max(keys)+1)
        self.featurizer = featurizer
    
    def predict(self, sentence: List[str]) -> int:
        fi = np.zeros(self.weights.shape)
        features = self.featurizer.extract_features(sentence, False)
        
        for k, v in features.items():
            fi[k] = v
            
        score = np.dot(self.weights, fi)
        
        #Sigmoid function
        y = np.exp(score) / (1+np.exp(score))
        return 1 if y > 0.5 else 0
    
    def update(self, lr, sentence, label):
        fi = np.zeros(self.weights.shape)
        features = self.featurizer.extract_features(sentence, False)
        
        for k, v in features.items():
            fi[k] = v
            
        score = np.dot(self.weights, fi)
        
        if label == 0:
            y = 1 / (1+np.exp(score))
            self.weights -= lr * fi * (1 - y)
        else:
            y = np.exp(score) / (1+np.exp(score))
            self.weights += lr * fi * (1 - y)
        


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # raise Exception("Must be implemented")
    lr = 0.1
    epochs = 30
    keys = []
    random.seed(24)
    
    for sentence in train_exs:
        counter = feat_extractor.extract_features(sentence.words, True)
        keys.extend(list(counter.keys()))
        
    model = PerceptronClassifier(list(set(keys)), feat_extractor)
    
    for epoch in range(1, epochs+1):
        if epoch % 10 == 0:
            lr /= 10
        random.shuffle(train_exs)
        for object in train_exs:
            sentence = object.words
            label = object.label
            pred = model.predict(sentence)
            if pred != label:
                model.update(lr, sentence, label)
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # raise Exception("Must be implemented")
    
    lr = 1
    epochs = 30
    keys = []
    random.seed(77)
    
    for sentence in train_exs:
        counter = feat_extractor.extract_features(sentence.words, True)
        keys.extend(list(counter.keys()))
    
    model = LogisticRegressionClassifier(list(set(keys)), feat_extractor)
    
    for epoch in range(1, epochs+1):
        if epoch % 10 == 0:
            lr /= 10
        
        random.shuffle(train_exs)
        for object in train_exs:
            sentence = object.words
            label = object.label
            pred = model.predict(sentence)
            
            if label != pred:
                model.update(lr, sentence, label)
    return model
def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
        feat_extractor.count_word_freq(train_exs)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model