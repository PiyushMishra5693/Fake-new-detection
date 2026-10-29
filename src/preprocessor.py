"""
Text preprocessing and normalization pipeline for news articles.
"""

import re
import string

# Standard curated English stopwords list (built-in so no heavy NLTK download is required)
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t',
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have',
    'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into',
    'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my',
    'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should',
    'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d',
    'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s',
    'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you',
    'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
}

CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"
}

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Cleans raw text for NLP model processing:
    - Removes URLs and email addresses
    - Expands common contractions
    - Strips non-alphanumeric punctuation (while preserving token boundaries)
    - Lowercases text
    - Optionally filters out common stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove HTML tags if present
    text = re.sub(r'<.*?>', ' ', text)
    
    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
        
    # Lowercase
    text = text.lower()
    
    # Remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize by whitespace
    tokens = text.split()
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
        
    return " ".join(tokens)

def combine_headline_and_body(title: str, text: str, headline_weight: int = 2) -> str:
    """
    Combines headline and text body, giving headline higher weight.
    """
    clean_title = clean_text(title)
    clean_body = clean_text(text)
    
    # Duplicate headline based on weight
    repeated_title = " ".join([clean_title] * max(1, headline_weight))
    return f"{repeated_title} {clean_body}".strip()
