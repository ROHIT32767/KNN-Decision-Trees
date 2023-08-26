import pickle
import os
import sys
import time
import numpy as np
import pandas as pd
from numpy import linalg as LA
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
        unique_values, counts = np.unique(np.array(A), return_counts=True)
        return unique_values[np.argmax(counts)]

    def get_distance(self, E1, E2):
        if self.metric == 'manhattan':
            return LA.norm(np.array(E1)-np.array(E2),axis=1,ord=1) # distance (x1,x2) -> |x1-x2|
        elif self.metric == 'euclidean':
            return LA.norm(np.array(E1)-np.array(E2),axis=1,ord=2)  # distance (x1,x2) -> |x1-x2|
        elif self.metric == 'cosine':
            return 1-(np.dot(np.array(E1),np.array(E2))) / (LA.norm(np.array(E1),axis=1) * LA.norm(np.array(E2)))
        else:
            raise ValueError("Unhandled Metric")
        
    def fit(self, train_embeddings, train_labels, validate_embeddings, validate_labels):
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.validate_embeddings = validate_embeddings
        self.validate_labels = validate_labels

    def data_split(self,data):
        self.labels = np.array(data[:, 3])
        self.embeddings = np.array(data[:,1:3])
        self.resnet = np.array([res[0] for res in np.array(data[:,1])])
        self.vit = np.array([v[0] for v in np.array(data[:,2])])
        num_total_samples = data.shape[0]
        num_training_samples = int(num_total_samples * self.ratio)
        indices = np.array(range(num_total_samples)) # used to check if unshuffled data is giving same results across users
        np.random.seed(42)
        indices = np.random.permutation(indices) # permutes the array [0,....n-1] 
        self.indices = indices # saving indices
        self.num_training_samples = num_training_samples # saving number of training samples

    def evaluate(self, embeddings, true_labels):
        predicted_labels = self.predict(embeddings)
        return f1_score(true_labels, predicted_labels, average='macro'),accuracy_score(true_labels, predicted_labels),precision_score(true_labels, predicted_labels, average='macro',zero_division=0), recall_score(true_labels, predicted_labels, average='macro',zero_division=0)

    def train(self, encoder_type):
        if encoder_type == 'vit':
            train_index = self.indices[:self.num_training_samples]
            validate_index = self.indices[self.num_training_samples:]
            self.fit(self.vit[train_index],self.labels[train_index],self.vit[validate_index],self.labels[validate_index])
        elif encoder_type == 'resnet':
            train_index = self.indices[:self.num_training_samples]
            validate_index = self.indices[self.num_training_samples:]
            self.fit(self.resnet[train_index],self.labels[train_index],self.resnet[validate_index],self.labels[validate_index])

    def predict_sample(self, E):
        distances = self.get_distance(self.train_embeddings, E)
        sorted_indices = np.argsort(distances)
        return self.get_measure(self.train_labels[sorted_indices[:self.k]])

    def predict(self, X):
        return np.array([self.predict_sample(embeddings) for embeddings in X])
    
    def print_answer(self,data):
        self.data_split(data)
        self.train(self.encoder_type)  
        F1_score, accuracy, precision, recall = self.evaluate(self.validate_embeddings, self.validate_labels)
        print(pd.DataFrame([['Accuracy',accuracy],['Precision',precision],['Recall',recall],['f1_score',F1_score]],columns=['Measure','Value']).to_string(index=False))
    
if len(sys.argv) != 2:
    print("Usage: python KNN.py <path_to_input_file>")
    sys.exit(1)

input_file = sys.argv[1]

if not os.path.isfile(input_file):
    print("Error: Input file '{}' not found.".format(input_file))
    sys.exit(1)


test_data = np.load(input_file, allow_pickle=True)
train_data = np.load('data.npy',allow_pickle=True)
Best_KNN_Object = Best_KNN(17,'vit','manhattan')
# Best_KNN_Object = Best_KNN(3,'vit','euclidean')
Best_KNN_Object.data_split(train_data)
Best_KNN_Object.train(Best_KNN_Object.encoder_type)
vit = np.array([v[0] for v in np.array(test_data[:,2])])
resnet = np.array([res[0] for res in np.array(test_data[:,1])])
if Best_KNN_Object.encoder_type == 'vit':
    test_embeddings = vit
elif Best_KNN_Object.encoder_type == 'resnet':
    test_embeddings = resnet
test_labels = np.array(test_data[:, 3])
F1_score, accuracy, precision, recall = Best_KNN_Object.evaluate(test_embeddings,test_labels)
print(pd.DataFrame([['Accuracy',accuracy],['Precision',precision],['Recall',recall],['f1_score',F1_score]],columns=['Measure','Value']).to_string(index=False))




