# utils/text_processor.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import re
import pandas as pd


# thesaurus = {
#     "break up": "break-up",
#     "make up": "make-up",
#     "move on": "move-on",
#     "get back": "get-back",
#     "hook up": "hook-up",
#     "settle down": "settle-down",
#     "house party": "house-party",
#     "move in": "move-in",
#     "friend zone": "friend-zone",
#     "date night": "date-night",
#     "long distance": "long-distance",
#     "heart break": "heartbreak",
#     "mother in law": "mother-in-law",
#     "father in law": "father-in-law",
#     "sister in law": "sister-in-law",
#     "brother in law": "brother-in-law",
#     "step father": "step-father",
#     "step mother": "step-mother",
#     "step sister": "step-sister",
#     "step brother": "step-brother",
#     "half sister": "half-sister",
#     "half brother": "half-brother",
#     "significant other": "significant-other",
#     "in laws": "in-laws",
#     "best friend": "best-friend",
#     "close friend": "close-friend",
#     "soul mate": "soul-mate",
#     "life partner": "life-partner",
#     "child care": "child-care",
#     "day care": "day-care",
#     "family member": "family-member",
#     "blood relation": "blood-relation",
#     "next of kin": "next-of-kin",
#     "gf": "girlfriend",
#     "bf": "boyfriend",
#     "so": "significant-other",
#     "girl friend": "girlfriend",
#     "boy friend": "boyfriend",
#     "safe space": "safe-space",
#     "mental health": "mental-health",
#     "self esteem": "self-esteem",
#     "self worth": "self-worth",
#     "self care": "self-care",
#     "self love": "self-love",
#     "self respect": "self-respect",
#     "self improvement": "self-improvement",
#     "self help": "self-help",
#     "self discovery": "self-discovery",
#     "self acceptance": "self-acceptance",
#     "self realization": "self-realization",
#     "self reflection": "self-reflection",
#     "self awareness": "self-awareness",
#     "self development": "self-development",
# }



def preprocess_text(text):
    """
    Clean and normalize text using NLTK.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)



    def standardize_relationship_terms(text):
        # Define a dictionary of terms to replace
        replacements = {
            # Partner-related terms
            r"\bgf\b": "girlfriend",
            r"\bbf\b": "boyfriend",
            r"\bso\b": "significant other",
            r"\bex-girlfriend\b": "ex girlfriend",
            r"\bex-boyfriend\b": "ex boyfriend",
            r"\bex-wife\b": "ex wife",

            # Family-related terms
            r"\bmom\b": "mother",
            r"\bdad\b": "father",
            r"\bgrandma\b": "grandmother",
            r"\bgrandpa\b": "grandfather",
            r"\bfam\b": "family",
            r"\bkids\b": "child",
            r"\bkid\b": "child",
            r"\bchildren\b": "child",
            r"\bparents\b": "parent",

            # Friend-related terms
            r"\bbuddies\b": "friend",
            r"\bbuddy\b": "friend",
            r"\bpals\b": "friend",
            r"\bpal\b": "friend",
            r"\bmates\b": "friend",
            r"\bmate\b": "friend",
            r"\bchums\b": "friend",
            r"\bchum\b": "friend",
            r"\ballies\b": "friend",
            r"\bally\b": "friend",
            r"\bacquaintances\b": "friend",
            r"\bacquaintance\b": "friend",
            r"\bcompanions\b": "friend",
            r"\bcompanion\b": "friend",
            r"\bfriends\b": "friend",
            r"\bfriends\b": "friend",
            r"\bcolleagues\b": "colleague",
            r"\bcoworkers\b": "coworker",
            r"\bteammates\b": "teammate",
            r"\broommates\b": "roommate"
        }
        
        # Replace each term in the text
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
            

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = standardize_relationship_terms(text)
    
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    extra_stop = ['tifu', 'aita', 'wibta','aitah',]
    stop_words.update(extra_stop)
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize based on POS tag
    lemmatizer = WordNetLemmatizer()
    tokens = pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(word, 'v') if tag.startswith('V')
        else lemmatizer.lemmatize(word)
        for word, tag in tokens
    ]

    # Remove short words
    tokens = [token for token in tokens if len(token) > 2 or token in ['gf', 'bf', 'ex', 'so']] # added "or....]"
    
    return ' '.join(tokens)

def split_label(label, max_line_length=25, max_lines=2):
    """Split label at the nearest space before max_line_length and return max_lines"""
    lines = []
    temp_label = label
    
    while len(temp_label) > max_line_length:
        split_index = temp_label.rfind(' ', 0, max_line_length)
        if split_index == -1:
            split_index = max_line_length
        lines.append(temp_label[:split_index])
        temp_label = temp_label[split_index:].strip()
        
    lines.append(temp_label)
    
    return '\n'.join(lines[:max_lines])