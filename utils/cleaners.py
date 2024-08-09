from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure that NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


async def clean_text(text: str, language: str) -> str:
    if language == 'en':
        stop_words = set(stopwords.words('english'))
    elif language == 'de':
        stop_words = set(stopwords.words('german'))
    else:
        stop_words = set()  # If language is not supported, don't remove stop words

    lemmatizer = WordNetLemmatizer() if language == 'en' else None
    tokens = word_tokenize(text.lower()) if language == 'en' else text.lower().split()
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ] if lemmatizer else [word for word in tokens if word.isalnum() and word not in stop_words]

    return ' '.join(cleaned_tokens)
