# utils/analysis.py
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text_processor import *
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS, TSNE
from utils.analysis import *
from scipy.spatial.distance import cdist
from adjustText import adjust_text
from scipy.stats import chi2_contingency 
import os

# Function to add any keyword columns /categorize the posts
def add_keyword_columns(df: pd.DataFrame, keywords: dict):
    ''' Add columns to the dataframe based on the keywords provided as a dictionary'''
    for category, pattern in keywords.items():
        df.loc[:, category] = df['selftext'].str.lower().str.contains(pattern)
    return df

# Define function to extract age and gender
# Handles both formats '37M' and 'M37' with 'I' preceding the term to ensure matches author
def extract_age_gender(text: str):
    
    # Handle missing values
    if pd.isna(text):
        return pd.Series([None, None])
    
    # Updated regex pattern to match both formats
    match = re.search(r"I.{0,3}\((?:(\d{2})(M|F|NB)|(M|F|NB)(\d{2}))\)", text)
    if match:
        # Determine the order based on which groups are matched
        if match.group(1) and match.group(2):  # Format: (age gender)
            age = int(match.group(1))
            gender = match.group(2)
        elif match.group(3) and match.group(4):  # Format: (gender age)
            gender = match.group(3)
            age = int(match.group(4))
        return pd.Series([age, gender])
    return pd.Series([None, None])

# Define function to get crosst tabulation of a column by subreddit and chi-square test
def analyze_column_by_subreddit(df, column_name):
    # crosstab for the specified column and subreddit
    crosstab = pd.crosstab(df['subreddit'], df[column_name], normalize='index') * 100
    # rename the columns using names dictionary
    names = {'AmItheAsshole': 'Am I the Asshole', 'confessions': 'Confessions', 'tifu': 'Today I Fucked Up'}
    # map values in subreddit column to names dictionary
    crosstab.index = crosstab.index.map(names)
    # remove column name in table
    crosstab.columns.name = None

    # save output to csv in figures directory
    if not os.path.exists('figures'):
        os.makedirs('figures')
    crosstab.to_csv(f'figures/{column_name}_by_subreddit.csv')

    # chi-square test for the specified column and subreddit
    crosstab2 = pd.crosstab(df['subreddit'], df[column_name])
    chi2, p, dof, expected = chi2_contingency(crosstab2)
    print(f"Chi-square: {chi2:.2f}")
    print(f"p-value: {p:.4f}")
    print(f"Degrees of freedom: {dof}")

    # show the crosstab
    return crosstab