import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spacy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """
    Preprocesses text by lowercasing, removing non-alphanumeric chars,
    tokenizing, removing stop words, and lemmatizing.

    Args:
        text (str): The input text.

    Returns:
        str: The processed text.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def calculate_tfidf(text_data, max_features=1000, ngram_range=(1, 2)):
    """
    Calculates TF-IDF scores for a list of text documents.

    Args:
        text_data (list): A list of processed text documents.
        max_features (int): Maximum number of features for TF-IDF.
        ngram_range (tuple): Range of n-grams to consider.

    Returns:
        tuple: tfidf_matrix, feature_names
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

def perform_lda_topic_modeling(tfidf_matrix, feature_names, num_topics=10):
    """
    Performs LDA topic modeling on the TF-IDF matrix.

    Args:
        tfidf_matrix (sparse matrix): The TF-IDF matrix.
        feature_names (list): The list of feature names.
        num_topics (int): The number of topics to discover.

    Returns:
        list: A list of top words for each topic.
    """
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)

    topics = []
    print(f'\nTopics discovered by LDA ({num_topics} topics):')
    for index, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        print(f'Topic #{index + 1}: {", ".join(top_words)}')
        topics.append(top_words)
    return topics

def perform_ner(text_data, num_samples=10):
    """
    Performs Named Entity Recognition on a sample of text data.

    Args:
        text_data (list): A list of text documents.
        num_samples (int): The number of samples to process.

    Returns:
        list: A list of entities for each processed document.
    """
    print('\nNamed Entities (using spaCy):\n')
    entities_list = []
    for i, text in enumerate(text_data[:num_samples]):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            print(f'Article {i + 1}: {entities}')
            entities_list.append(entities)
        else:
            print(f'Article {i + 1}: No named entities found.')
            entities_list.append([])
    return entities_list