#! /usr/bin/env python3

import argparse
import os
import numpy as np
import keras
from datahandler.data_loader import DataLoader
from datahandler.data_preprocessing import DataPreprocessing
from representation.representation_handler import RepresentationHandler
from representation.classifier import Classifier

#This function read and parse the args
def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--operation', type=str, choices=["train", "execution"],
            help='The execution method: <train> or <execution>', required=True)
    parser.add_argument('--path', type=str, help="path to directory or to the\
            review file", required=True)
    return parser.parse_args()

#This function verifies if the input file or directory exists
def check_path(path):
    if os.path.exists(path):
        return True
    else:
        print("File or directory path does not exists")
        exit()

#This function instantiate the classifier class for a training execution
def train_classifier(reviews_vectors, train_labels, test_labels):
    train = np.array(reviews_vectors[:25000])
    test = np.array(reviews_vectors[25000:])
    y_train = keras.utils.to_categorical(train_labels, num_classes=2)
    y_test = keras.utils.to_categorical(test_labels, num_classes=2)
    classifier = Classifier()
    classifier.create_mlp_model()
    classifier.train(train, y_train)
    classifier.evaluate(test, y_test)
    scores = classifier.get_score()
    print("Score MLP " + str(scores["mlp"]))
    classifier.save()

#This function instantiate the classifier class for a execution with
#a trained model
def test_classifier(review):
    classifier = Classifier()
    classifier.load()
    mlp_pred = classifier.get_prediction(review)
    #this is to print according to the output requirements
    #neg=0 but we print -1
    if mlp_pred == [0]: 
        mlp_pred = -1
    else:
        mlp_pred = 1
    print(mlp_pred)

def main():
    args = read_args()
    operation = args.operation
    if check_path(args.path): path = args.path
    data_loader = DataLoader(operation, path)
    data_loader.load_data()
    data = data_loader.get_data()

    dt = DataPreprocessing(operation)
    representation_model = RepresentationHandler(operation)
    if operation == "train":
        tr_rew, train_labels, ts_rew, test_labels = dt.pre_process_data(data)
        reviews_vectors = representation_model.build(tr_rew+ts_rew)
        train_classifier(reviews_vectors, train_labels, test_labels)
    else:
        review = dt.pre_process_data(data)
        review_vector = representation_model.build(review)
        test_classifier(np.array(review_vector))

if __name__ == "__main__":
    main()
