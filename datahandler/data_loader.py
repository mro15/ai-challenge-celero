#! /usr/bin/env python3

class DataLoader(object):
    def __init__(self, operation, path):
        self.operation = operation
        self.data = None
        self.labels = None
        self.path = path

    def load_data(self):
        self.data = self.read_train_data() if self.operation == "train" else self.read_execution_data()

    def read_train_data(self):
        reviews = {"pos": [], "neg": []}
        return reviews

    def read_execution_data(self):
        review = []
        return review
