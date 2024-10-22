import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymystem3

class TextPreprocessor:
    def __init__(self, language='russian'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = pymystem3.Mystem()  

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        
        try:
            # Удаление персональных данных
            text = re.sub(r'\d{4,}', '[НЗ]', text)
            text = re.sub(r'[A-Za-z]\d{6,}', '[НК]', text)
            text = re.sub(r'\S+@\S+', '[EMAIL]', text)
            text = re.sub(r'http\S+|www\S+', '[URL]', text)
            text = re.sub(r'\+?\d[\d -]{8,12}\d', '[PHONE]', text)
            text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '[DATE]', text)
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
            # Нормализация текста
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
        
            # Удаление стоп-слов и лемматизация
            words = word_tokenize(text)
            filtered_words = [self.lemmatizer.lemmatize(word)[0] for word in words if word.lower() not in self.stop_words]
            
            if not filtered_words:
                raise ValueError("Пустой словарный запас; возможно, документы содержат только стоп-слова")
            
            text = ' '.join(filtered_words)
        
            # Удаление повторяющихся символов
            text = re.sub(r'(.)\1{2,}', r'\1', text)
        
            return text
        except ValueError as e:
            print(f"Ошибка: {e}")
            return ""
        
class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def vectorize_text(self, text_list):
        X = self.vectorizer.fit_transform(text_list)
        return X, self.vectorizer