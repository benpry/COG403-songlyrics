"""
This file reads artists from the file "artists.csv",
determines their gender using a Naive Bayes Classifier trained on the US National Names dataset, 
and saves the names of solo artists with their gender in the file "genders.csv"
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import csv

NAMES_PATH = "../../data/NationalNames.csv"
ARTISTS_PATH = "../../data/processed/artists.csv"
OUTPUT_PATH = "../../data/processed/artist_genders.csv"

if __name__ == "__main__":

    # Get US National Names data
    data = pd.read_csv(NAMES_PATH, dtype = {'Count': np.int32})
    # Group data by name and gender
    grouped_data = data.groupby(['Name', 'Gender'], as_index = False)['Count'].sum()
    # Categorize names as male or female based on frequency
    gendered_data = grouped_data.reset_index().pivot('Name', 'Gender', 'Count')
    gendered_data = gendered_data.fillna(0)
    gendered_data['Gender'] = np.where(gendered_data['F'] > gendered_data['M'], 'female', 'male')
    # Use char vectorizer matrix X of name by 2 character n grams
    # Rows are names, columns are ngrams, elements are (name_i, ngram_j) for ith name, jth ngram
    char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X = char_vectorizer.fit_transform(gendered_data.index)
    X = X.tocsc()
    # y is a list of values to corresponding names
    # Female names as marked as 1 and male names as 0
    y = (gendered_data.Gender == 'female').values.astype(int)
    # Split our training and test data now
    itrain, itest = train_test_split(range(gendered_data.shape[0]), train_size=0.7)
    mask = np.ones(gendered_data.shape[0], dtype='int')
    mask[itrain] = 1
    mask[itest] = 0
    mask = (mask==1)
    # Train the model
    Xtrainthis = X[mask]
    Ytrainthis = y[mask]
    Xtestthis = X[~mask]
    Ytestthis = y[~mask]
    clf = MultinomialNB(alpha = 1)
    clf.fit(Xtrainthis, Ytrainthis)
    # training_accuracy = clf.score(Xtrainthis,Ytrainthis)
    # test_accuracy = clf.score(Xtestthis,Ytestthis)      
    # print("Training Accuracy: ", training_accuracy)
    # print("Test Accuracy:", test_accuracy)

    # Lookup function to get predicted gender of a name
    def lookup(x):
        str(x)
        new = char_vectorizer.transform([x])
        y_pred = clf.predict(new)
        if (y_pred == 1):
            return "female"
        else:
            return "male"
    

    # Get all artists 
    artists = []
    fp = open(ARTISTS_PATH, "r", encoding="utf-8")
    lines = fp.readlines()
    artists = [line.rstrip('\n') for line in lines]
    
    # Get genders
    artist_genders = []
    for artist in artists:
        artist_genders.append([artist, lookup(artist)])
    
    # Write to file
    with open(OUTPUT_PATH, 'w') as outfile:
        mywriter = csv.writer(outfile)
        for i in artist_genders:
            mywriter.writerow(i)