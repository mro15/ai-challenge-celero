#! /usr/bin/env python3

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.models.doc2vec
import multiprocessing

"""
    This class works with the feature extration. For text represesentation
    the paragraph vector model is used.
"""
class RepresentationHandler(object):
    def __init__(self, operation):
        self.model = None
        self.operation = operation
        self.model_path = "representation/doc2vecmodel"
        self.cores = multiprocessing.cpu_count()

    """
        This function see the program operation mode and according to it calls
        the function to create, train and save the doc2vec model or to load an
        alread created and trained model
    """
    def build(self, data):
        if self.operation == "train":
            self.create_and_train_model(data)
            self.save_model()
            #self.load_model()
            return self.retrieve_reviews_vectors(data)
        else:
            self.load_model()
            return self.infer_new_review(data)
            
    """
        This function instantiate the doc2vec model and train it
    """
    def create_and_train_model(self, reviews):
        assert gensim.models.doc2vec.FAST_VERSION > -1
        #for each review (document), associate a tag number
        docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews)]
        #instantiate the doc2vec model
        self.model = Doc2Vec(vector_size=100, window=10, min_count=1, workers=self.cores,
                            epochs=50, sample=1e-3, negative=5, dm=0)
        self.model.build_vocab(docs) #build vocabulary
        self.model.train(docs, total_examples=self.model.corpus_count,
                        epochs=self.model.epochs) #train the model

    """
        This function saves the trained doc2vec model
    """
    def save_model(self):
        self.model.save(self.model_path)

    """
        This function loads the trained doc2vec model
    """
    def load_model(self):
        self.model = Doc2Vec.load(self.model_path)

    """
        This function receives a new review as parameter and infer his
        representation based on the already trained doc2vec model
    """
    def infer_new_review(self, review):
        return  self.model.infer_vector(review)

    """
        This function returns the representation (vector) of each review
    """
    def retrieve_reviews_vectors(self, reviews):
        reviews_vector = [self.model.docvecs[i] for i, doc in enumerate(reviews)]
        return reviews_vector

