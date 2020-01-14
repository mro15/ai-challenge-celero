#! /usr/bin/env python3

import nltk

"""
    This class works with the data pre processing in order to avoid data noise.
"""
class DataPreprocessing(object):
    def __init__(self, operation):
        self.operation = operation
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stem = nltk.stem.PorterStemmer()
        nltk.download('stopwords')
        self.stop_words = set(nltk.corpus.stopwords.words('english')) 

    """
        This function see the program operation mode and according to it calls
        the function to pre process the input data
    """
    def pre_process_data(self, data):
        if self.operation == "train": 
            train_reviews, train_labels = self.split_data_in_arrays(data, "train")
            test_reviews, test_labels = self.split_data_in_arrays(data, "test")
            for i in range(0, len(train_reviews)):
                train_reviews[i] = self.pre_process(train_reviews[i])
            for i in range(0, len(test_reviews)):
                test_reviews[i] = self.pre_process(test_reviews[i])
            return train_reviews, train_labels, test_reviews, test_labels
        else:
            return self.pre_process(data)

    """
        This function returns an array with the data in the dict and also an array
        with the corresponding labels
        The parameter "d_type" indicates if the data to be retrieved is train or
        test
    """
    def split_data_in_arrays(self, data, d_type):
        vec_data =  data[d_type]["neg"] + data[d_type]["pos"]
        labels = [-1]*len(data[d_type]["neg"]) + [1]*len(data[d_type]["pos"])
        return vec_data, labels

    """"
        This function does the pre processing of a single review.
        The pre processing steps includes:
        Convert to lowercase, tokenize, remove non-alphabetic characteres,
        remove stop words and stemming
    """
    def pre_process(self, review):
        tokens = review.lower()
        tokens = self.tokenizer.tokenize(tokens) #tokenize the review
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [w for w in tokens if not w in self.stop_words]
        tokens = [self.stem.stem(w) for w in tokens]
        return tokens
 
