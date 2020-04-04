from read_data import read_data
from feature_extraction import feature_extractor
from create_model import create
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fs',default = 600, type = int, help = 'Sampling Frequency')
    parser.add_argument('--wlen',default =4, type = int, help = 'Window Length in seconds')
    parser.add_argument('--overlap',default = 0.25, type = int, help = 'Fraction of overlap (default = 0.25)')
    
    args = parser.parse_args()

    read_data()
    feature_extractor(args)
    create()