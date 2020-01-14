#! /usr/bin/env python3

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn.utils import shuffle
import numpy as np


"""
    This class works with the data classification, a Multi Layer Perceptron
    is the classifier used.
"""
class Classifier(object):
    def __init__(self):
        self.mlp_model = None
        self.score = {}
        self.score["mlp"] = 0
        self.mlp_save = "representation/mlp_model"

    """
        This function create a sequantial model using keras, in the end of the
        network we stack a softmax layer to classify the examples
    """
    def create_mlp_model(self):
        model = Sequential()
        model.add(Dense(200, activation='relu', input_dim=100))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        self.mlp_model = model

    """
        This function train the classifier using the train dataset part
    """
    def train(self, train_data, train_labels):
        #shuffle data to avoid overfitting
        train_data, train_labels = shuffle(train_data, train_labels, random_state=0)
        self.mlp_model.fit(train_data, train_labels, validation_split=0.25, epochs=64, batch_size=32)

    """
        This function saves the trained model
    """
    def save(self):
        self.mlp_model.save(self.mlp_save)

    """
        This function loads the trained model
    """
    def load(self):
        self.mlp_model = load_model(self.mlp_save)

    """
        This function evaluates the classifier in the test dataser part.
        The evaluation metric used is the classifier accuracy
    """
    def evaluate(self, test_data, test_labels):
        #shuffle test data
        test_data, test_labels = shuffle(test_data, test_labels, random_state=0)
        mlp_score = self.mlp_model.evaluate(test_data, test_labels)
        self.score["mlp"] = mlp_score[-1]

    """
        This function return the classifier score (accuracy)
    """
    def get_score(self):
        return self.score

    """
        This function returns the classifier prediction of a unseen review
        in the execution operation of the program
    """
    def get_prediction(self, review):
        review = review.reshape(1, -1)
        mlp_pred = self.mlp_model.predict_classes(review) 
        return mlp_pred
