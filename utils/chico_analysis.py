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


def tfidf_df(posts_df, max_terms=1000, min_doc_freq=2, include_selftext=False):
    """
    Analyze a single subreddit's posts independently.
    """
    # Combine title and optionally selftext
    # preprocess_text(post.get('title', '')) + (' ' + preprocess_text(post.get('selftext', '')) if include_selftext else '')
    texts = [f"{preprocess_text(row['title'])} {preprocess_text(row['selftext'])}" for _, row in posts_df.iterrows()]


    # Analyze vocabulary first
    freq_df, vocab_stats = analyze_vocabulary(texts, min_freq=min_doc_freq)
    # Generate TF-IDF matrix and feature names
    tfidf_matrix, feature_names = generate_tfidf_matrix(texts, max_terms, min_doc_freq)
    
    # Create results object from the matrix and feature names
    results = {
        "tfidf_matrix": tfidf_matrix, 
        "feature_names": feature_names, 
        "freq_df":freq_df, 
        "vocab_stats":vocab_stats}
    
    return results

def plot_word_distance(tfidf_matrix, feature_names, word, distance, n_highlight=5, perplexity=30, title=None):
    """
    Plot word similarities using t-SNE with all terms, filtering only words within a specified distances
    from a given word.
    """
    # Get vectors for all terms
    term_vectors = tfidf_matrix.T.toarray()
    
    # Calculate t-SNE for all terms
    tsne = TSNE(n_components=2, 
                perplexity=min(perplexity, len(feature_names) / 4), 
                random_state=42)
    coords = tsne.fit_transform(term_vectors)
    
    # Identify the index and coordinates of the origin word
    try:
        word_index = list(feature_names).index(word)
    except ValueError:
        print(f"Word '{word}' not found in the feature list.")
        return
    
    origin_coords = coords[word_index]
    
    # Calculate distances from the origin word to all other words
    distances = cdist([origin_coords], coords, metric='euclidean')[0]
    
    # Identify terms within the specified distance
    within_distance_indices = [i for i, d in enumerate(distances) if d <= distance]
    
    # Calculate relevance (mean TF-IDF score) for terms within the distance
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    relevant_terms = [(i, mean_tfidf[i]) for i in within_distance_indices]
    
    # Sort by relevance and select top n_highlight terms
    relevant_terms = sorted(relevant_terms, key=lambda x: x[1], reverse=True)[:n_highlight]
    top_indices_within_distance = [i for i, _ in relevant_terms]
    top_terms_within_distance = [feature_names[i] for i in top_indices_within_distance]
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot all points within the distance threshold in light gray, highlighting top terms in red
    for i in within_distance_indices:
        color = 'red' if i in top_indices_within_distance else 'lightgray'
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=30)
        
    # Highlight the origin word at its original position
    ax.scatter(origin_coords[0], origin_coords[1], s=100, color='blue')  # Origin word
    ax.annotate(word, (origin_coords[0], origin_coords[1]), fontsize=14,
                bbox=dict(facecolor='white', edgecolor='blue', alpha=0.7))
    
    # Prepare labels for adjustment
    texts = []
    for i in top_indices_within_distance:
        if i != word_index:  # Exclude the origin word
            texts.append(ax.text(coords[i, 0], coords[i, 1], feature_names[i],
                                fontsize=14,
                                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7)))
 
    # Adjust text to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Set the title
    if title:
        ax.set_title(f'Word Similarities in {title} (Words within {distance} of "{word}")')
    else:
        ax.set_title(f'Word Similarities (Words within {distance} of "{word}")')
    
    # Adjust layout and return fig and ax
    plt.tight_layout()
    return fig, ax


def cosine_dist_word_net(tf_idf_matrix, feature_names, word, distance):
    """
    Calculate cosine distances between two documents in a concize space defined by the euclidean distance.
    """
    # Get vectors for all terms
    term_vectors = tf_idf_matrix.T.toarray()
    
    # Calculate t-SNE for all terms
    tsne = TSNE(n_components=2, 
                perplexity=min(perplexity, len(feature_names) / 4), 
                random_state=42)
    coords = tsne.fit_transform(term_vectors)
    
    # Identify the index and coordinates of the origin word
    try:
        word_index = list(feature_names).index(word)
    except ValueError:
        print(f"Word '{word}' not found in the feature list.")
        return
    
    origin_coords = coords[word_index]
    
    # Calculate distances from the origin word to all other words
    distances = cdist([origin_coords], coords, metric='euclidean')[0]
    
    # Identify terms within the specified distance
    within_distance_indices = [i for i, d in enumerate(distances) if d <= distance]

    # Calculate relevance (mean TF-IDF score) for terms within the distance
    mean_tfidf = tf_idf_matrix.mean(axis=0).A1
    relevant_terms = [(i, mean_tfidf[i]) for i in within_distance_indices]
    
    # Sort by relevance and select top n_highlight terms
    relevant_terms = sorted(relevant_terms, key=lambda x: x[1], reverse=True)[:n_highlight]

    return relevant_terms

def cosine_between_doc(full_result):
    for sub in full_result:
        
