import re
import string
import logging
import nltk

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
# --- Added Block ---
try:
    # PunktTokenizer seems to sometimes need this implicitly
    nltk.data.find('tokenizers/punkt_tab')
    logging.info("NLTK 'punkt_tab' resource already available.")
except LookupError:
    logging.info("Downloading NLTK 'punkt_tab' resource...")
    nltk.download('punkt_tab', quiet=True)
# --- End Added Block ---

import Stemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Preprocessor:
    def __init__(self, remove_stopwords=True, enable_stemming=True):
        self.remove_stopwords = remove_stopwords
        self.enable_stemming = enable_stemming
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = Stemmer.Stemmer("english") if enable_stemming else None
        self.punctuation_table = str.maketrans('', '', string.punctuation)
        stemmer_type = 'PyStemmer (English Snowball)' if enable_stemming else 'Disabled'
        logging.info(f"Preprocessor initialized (Stopwords: {remove_stopwords}, Stemming: {stemmer_type})")

    def get_stemmer(self):
        return self.stemmer

    def clean_text(self, text):
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
        if not text:
            return []
        try:
            text = self.clean_text(text)
            tokens = word_tokenize(text)
        except Exception as e:
            logging.error(f"Error during word_tokenize for text snippet '{text[:50]}...': {e}")
            return []

        processed_tokens = []
        for token in tokens:
            if not token: continue
            if self.remove_stopwords and token in self.stop_words:
                continue
            if self.stemmer:
                token = self.stemmer.stemWord(token)
            processed_tokens.append(token)
        return processed_tokens

    def preprocess_for_bert(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text