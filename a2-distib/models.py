# models.py

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(300, 200)
        # self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 100)
        # self.dropout1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))        
        return x
        
        
    

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model, embedding):
        # raise NotImplementedError
        self.model = model
        self.embedding = embedding
        
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        word_sum = np.zeros(300)
        for i in range(len(ex_words)):
            word_embedding = self.embedding.get_embedding(ex_words[i])
            word_sum += word_embedding
            
        # word_average is a List
        word_average = word_sum / len(ex_words)
        train_x_batch = torch.unsqueeze(torch.from_numpy(word_average).float(), 0)

        # train_x_batch = torch.from_numpy(word_average).float()
        # output = torch.argmax(self.model.forward(train_x_batch))
        output = torch.argmax(self.model.forward(train_x_batch)[0])
        
        return output
    
    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]

def sentiment2embed(train_exs, word_embeddings):
    vector, labels = [], []
    
    for i in range(len(train_exs)):
        word_sum = np.zeros(300)
        for j in range(len(train_exs[i].words)):
            embedding = word_embeddings.get_embedding(train_exs[i].words[j])
            word_sum += np.array(embedding)
            
        word_average = word_sum / len(train_exs)
        vector.append(word_average)
        
        label = train_exs[i].label
        labels.append(label)
        
    return np.array(vector), np.array(labels)
        
    


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    # raise NotImplementedError
    
    train_x, train_y = sentiment2embed(train_exs, word_embeddings)
    epochs = 30
    batch_size = args.batch_size
    fnn = FNN()
    lr = 0.0001
    optimizer = optim.Adam(fnn.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    fnn.train(True)
    
    for epoch in range(epochs):
        n = len(train_x) // batch_size
        training_loss = 0.0
        training_accuracy = 0.0
        ex_indices = [i for i in range(0, len(train_x))]
        random.shuffle(ex_indices)
        train_x = train_x[ex_indices, ...]
        train_y = train_y[ex_indices, ...]
        
        
        for i in range(n):
            train_x_batch = torch.from_numpy(train_x[i:i + batch_size, :]).float()
            train_y_batch = torch.from_numpy(train_y[i:i + batch_size])
            
            optimizer.zero_grad()
            output = fnn(train_x_batch)
            
            loss = loss_function(output, train_y_batch)
            training_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        
        epoch_loss = training_loss/n
        print("Total loss on epoch %i: %f" % (epoch, epoch_loss))
        
    return NeuralSentimentClassifier(fnn, word_embeddings)

