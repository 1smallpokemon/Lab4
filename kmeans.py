import sys
import pandas as pd
from preprocessing import *

def main():
    # call preprocessing script, returns dataframe
    argc = len(sys.argv)
    
    if argc < 3:
        print("Usage: python kmeans.py <Filename> <k>")
    
    try:
        uncleaned_data =  load_data(sys.argv[1])
        data =  preprocess_data(uncleaned_data)
        
        k = sys.argv[2]
    except:
        print("Couldn't parse command line")
        
    
    pass





if __name__ == "__main__":
    main()
    
    