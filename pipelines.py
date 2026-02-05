import pandas as pd


def preprocess_college_data(url):
    # Read in the data set
    college=pd.read_csv(url)

    