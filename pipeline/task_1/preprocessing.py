import re
import string
import logging
import nltk

# Ensure required NLTK resources are available, download if missing
try:
    nltk.data.find('tokenizers/punkt')
    logging.info("NLTK 'punkt' tokenizer already available.")
except LookupError:
    logging.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
    logging.info("NLTK 'stopwords' already available.")
except LookupError:
    logging.info("Downloading NLTK 'stopwords'...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
    logging.info("NLTK 'punkt_tab' resource already available.")
except LookupError:
    logging.info("Downloading NLTK 'punkt_tab' resource...")
    nltk.download('punkt_tab', quiet=True)

import Stemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Preprocessor:
    """
    Text preprocessing class for IR pipeline.
    Supports lowercasing, punctuation removal, stopword removal, stemming, and tokenization.
    """    
    def __init__(self, remove_stopwords=True, enable_stemming=True):
        """
        Initialize Preprocessor.

        Args:
            remove_stopwords (bool): Whether to remove English stopwords.
            enable_stemming (bool): Whether to apply stemming using PyStemmer.
        """
        self.remove_stopwords = remove_stopwords
        self.enable_stemming = enable_stemming
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = Stemmer.Stemmer("english") if enable_stemming else None
        self.punctuation_table = str.maketrans('', '', string.punctuation)
        stemmer_type = 'PyStemmer (English Snowball)' if enable_stemming else 'Disabled'
        logging.info(f"Preprocessor initialized (Stopwords: {remove_stopwords}, Stemming: {stemmer_type})")

    def get_stemmer(self):
        """
        Returns the stemmer instance (if enabled).
        """        
        return self.stemmer

    def clean_text(self, text):
        """
        Clean the input text by lowercasing, removing punctuation, 
        standalone digits, single characters and extra spaces.

        Args:
            text (str): Input text.

        Returns:
            str: Cleaned text.
        """        
        if not text:
            return ""
        text = text.lower()
        text = text.translate(self.punctuation_table)
        # text = re.sub(r'\W+', ' ', text) # no improvement
        # text = re.sub(r'\d+', '', text)  # Remove all digits -> don't do that.
        text = re.sub(r'\b\d+\b', '', text)  # Remove stand alone digits
        text = re.sub(r'\b\w{1}\b', '', text)  # Remove single characters
        text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces
        return text

    def tokenize_and_process(self, text):
        """
        Tokenize the input text, remove stopwords and apply stemming if enabled.

        Args:
            text (str): Input text.

        Returns:
            list: List of processed tokens.
        """        
        if not text:
            return []
        try:
            text = self.clean_text(text)    # use the cleaned text 
            tokens = word_tokenize(text)    # create word tokens
        except Exception as e:
            logging.error(f"Error during word_tokenize for text snippet '{text[:50]}...': {e}")
            return []

        processed_tokens = []
        for token in tokens:
            if not token: continue
            if self.remove_stopwords and token in self.stop_words:  # first remove stopwords...
                continue
            if self.stemmer:
                token = self.stemmer.stemWord(token)    # ...then stem the token
            processed_tokens.append(token)
        return processed_tokens

