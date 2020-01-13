#! /usr/bin/env python3

import argparse
from datahandler.data_loader import DataLoader
import os

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--operation', type=str, choices=["train", "execution"], help='The execution method: <train> or <execution>', required=True)
    parser.add_argument('--path', type=str, help="path to directory or to the review file", required=True)
    return parser.parse_args()

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
    data = data_loader.load_data()


if __name__ == "__main__":
    main()
