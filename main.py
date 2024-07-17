from preprocessing.preprocessing import TextPreprocessor, TextVectorizer

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