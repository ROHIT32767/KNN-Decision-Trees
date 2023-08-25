import pickle
import os
import sys
import time
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
class Best_KNN:
    def __init__(self, k, encoder_type, metric,ratio=0.8):
        self.k = k
        self.encoder_type = encoder_type
        self.metric = metric
        self.measure = 'mode'
        self.ratio = ratio

    def get_encoder_type(self):
        return self.encoder_type
    def set_encoder_type(self,encoder_type):
        self.encoder_type = encoder_type

    def get_metric(self):
        return self.metric
    def set_metric(self,metric):
        self.metric = metric

    def get_k(self): # get method for K
        return self.k
    def set_k(self,k): # set method for K
        self.k = k

    def get_ratio(self):
        return self.ratio
    def set_ratio(self,ratio):
        self.ratio = ratio

    def get_measure(self, A):
        if self.measure == 'mode':
            unique_values, counts = np.unique(np.array(A), return_counts=True)
            mode_index = np.argmax(counts)
            mode = unique_values[mode_index]
            return mode

    def get_distance(self, E1, E2):
        if self.metric == 'manhattan':
            return norm(np.array(E1)-np.array(E2),axis=1,ord=1)
        elif self.metric == 'euclidean':
            return norm(np.array(E1)-np.array(E2),axis=1,ord=2)
        elif self.metric == 'cosine':
            return 1-(np.dot(np.array(E1),np.array(E2))) / (norm(np.array(E1),axis=1) * norm(np.array(E2)))
        else:
            raise ValueError("Invalid metric")
        
    def fit(self, train_embeddings, train_labels, validate_embeddings, validate_labels):
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.validate_embeddings = validate_embeddings
        self.validate_labels = validate_labels

    def data_split(self,data):
        self.labels = np.array(data[:, 3])
        self.embeddings = np.array(data[:,1:3])
        resnet = np.array(data[:,1])
        self.resnet = [res[0] for res in resnet]
        self.resnet = np.array(self.resnet)
        vit = np.array(data[:,2])
        self.vit = [v[0] for v in vit]
        self.vit = np.array(self.vit)
        num_total_samples = data.shape[0]
        num_training_samples = int(num_total_samples * self.ratio)
        indices = np.array(range(num_total_samples)) # used to check if unshuffled data is giving same results across users
        np.random.shuffle(indices) # permutes the array [0,....n-1] 
        self.indices = indices # saving indices
        self.num_training_samples = num_training_samples # saving number of training samples

    def evaluate(self, embeddings, true_labels):
        predicted_labels = self.predict_array(embeddings)
        F1_score = f1_score(true_labels, predicted_labels, average='macro')
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='macro',zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='macro',zero_division=0)
        return F1_score, accuracy, precision, recall

    def train(self, encoder_type):
        if encoder_type == 'vit':
            train_index = self.indices[:self.num_training_samples]
            validate_index = self.indices[self.num_training_samples:]
            data_train = self.vit[train_index]
            data_validate = self.vit[validate_index]
            label_train = self.labels[train_index]
            label_validate = self.labels[validate_index]
            self.fit(data_train, label_train, data_validate, label_validate)
        elif encoder_type == 'resnet':
            train_index = self.indices[:self.num_training_samples]
            validate_index = self.indices[self.num_training_samples:]
            data_train = self.resnet[train_index]
            data_validate = self.resnet[validate_index]
            label_train = self.labels[train_index]
            label_validate = self.labels[validate_index]
            self.fit(data_train, label_train, data_validate, label_validate)

    def predict_sample(self, E):
        distances = self.get_distance(self.train_embeddings, E)
        sorted_indices = np.argsort(distances)
        nearest_index = sorted_indices[:self.k]
        nearest_labels = self.train_labels[nearest_index]
        classified_label = self.get_measure(nearest_labels)
        return classified_label

    def predict_array(self, X):
        predictions = [self.predict_sample(embeddings) for embeddings in X]
        return np.array(predictions)
    
    def print_answer(self,data):
        self.data_split(data)
        self.train(self.encoder_type)  
        F1_score, accuracy, precision, recall = self.evaluate(self.validate_embeddings, self.validate_labels)
        print_data = [['Accuracy',accuracy],['Precision',precision],['Recall',recall],['f1_score',F1_score]]
        df = pd.DataFrame(print_data,columns=['Measure','Value'])
        print(df.to_string(index=False))
    
if len(sys.argv) != 2:
    print("Usage: python KNN.py <path_to_input_file>")
    sys.exit(1)

input_file = sys.argv[1]

if not os.path.isfile(input_file):
    print("Error: Input file '{}' not found.".format(input_file))
    sys.exit(1)

