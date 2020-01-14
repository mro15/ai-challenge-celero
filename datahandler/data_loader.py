#! /usr/bin/env python3

from os import listdir

"""
    This class works with the data (movie reviews). It has functions for
    loading data according to the program's operating mode (train or execution).
"""
class DataLoader(object):
    def __init__(self, operation, path):
        self.operation = operation
        self.data = None
        self.path = path

    """
        This method returns the data attribute, with the stored review(s)
    """
    def get_data(self):
        return self.data

    """
        This function see the program operation mode and according to it calls
        the function to read the input data
    """
    def load_data(self):
        if self.operation == "train": 
            self.data = self.read_train_data() 
        else:
            self.data = self.read_execution_data()

    """
        This function will read and return the data in a directory for a
        training operation mode of the program
    """
    def read_train_data(self):
        #use a dict to store the reviews separating between train and test with
        #positive and negative labels
        reviews = {}
        reviews["train"] = {"pos": [], "neg": []}
        reviews["test"] = {"pos": [], "neg": []}

        for i in ["train", "test"]:
            for j in ["pos", "neg"]:
                directory = self.path+i+"/"+j+"/"
                files = listdir(directory) #list file in directory
                for f in files:
                    #read the review and store it in the appropriate dict position
                    fp = open(directory+f, 'r')
                    reviews[i][j].append(fp.read())
        return reviews

    """
        This function will read and return the data in a file for an
        execution operation mode of the program
    """
    def read_execution_data(self):
        fp = open(self.path, 'r')
        review = fp.read()
        return review
