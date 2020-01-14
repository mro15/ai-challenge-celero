#! /usr/bin/env python3

import argparse
from datahandler.data_loader import DataLoader
from datahandler.data_preprocessing import DataPreprocessing
import os


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

def main():
    args = read_args()
    operation = args.operation
    if check_path(args.path): path = args.path
    data_loader = DataLoader(operation, path)
    data_loader.load_data()
    data = data_loader.get_data()

    pre_process = DataPreprocessing(operation)
    if operation == "train":
        train_reviews, train_labels, test_reviews, test_labels = pre_process.pre_process_data(data)
    else:
        review = pre_process.pre_process_data(data)

if __name__ == "__main__":
    main()
