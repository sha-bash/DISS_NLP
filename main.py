from preprocessing.preprocessing import TextPreprocessor, TextVectorizer
from ML.ClassicML import LogisticRegressionModel, SVMModel, RandomForestModel
from sklearn.model_selection import train_test_split


texts = [...]  
labels = [...]  

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Предобработка текстов
preprocessor = TextPreprocessor(language='russian')
X_train = [preprocessor.preprocess_text(text) for text in X_train]
X_test = [preprocessor.preprocess_text(text) for text in X_test]

# Векторизация текстов
vectorizer = TextVectorizer()
X_train_vec, _ = vectorizer.vectorize_text(X_train)
X_test_vec, _ = vectorizer.vectorize_text(X_test)

# Обучение и оценка моделей
models = [
    LogisticRegressionModel(),
    SVMModel(),
    RandomForestModel()
]

for model in models:
    print(f"Training and evaluating {model.__class__.__name__}")
    model.train(X_train_vec, y_train)
    model.evaluate(X_test_vec, y_test)
    print("\n" + "="*50 + "\n")

    
if __name__ == "__main__":
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    text = input('Введите текст: ')
    try:
        preprocessor = TextPreprocessor(language='russian')
        preprocessed_text = preprocessor.preprocess_text(text)
        print("Обработанный текст:", preprocessed_text)
    
        vectorizer = TextVectorizer()
        vectorized_text, vectorizer = vectorizer.vectorize_text([preprocessed_text])
        print("Векторизованный текст:", vectorized_text.toarray())
        print("Признаки:", vectorizer.get_feature_names_out())
    except ValueError as e:
        print(e)