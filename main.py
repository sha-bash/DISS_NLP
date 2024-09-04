import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils.config import get_connection
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.preprocessing import TextPreprocessor, TextVectorizer
from ML.ClassicML import LogisticRegressionModel, SVMModel, RandomForestModel
from ML.VisualizationModel import VisualizationModel  # Предполагается, что класс определен в отдельном файле
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка необходимых ресурсов для NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Загрузка данных из базы данных
conn = get_connection()
db_config = {
    'dbname': conn.info.dbname,
    'user': conn.info.user,
    'password': conn.info.password,
    'host': conn.info.host,
    'port': conn.info.port
}

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT processed_text, label_id FROM processed_texts")
    data = cursor.fetchall()
    texts = [row[0] for row in data if row[0] != '']  # Удаление пустых строк
    labels = [row[1] for row in data if row[0] != '']  # Удаление соответствующих меток
    cursor.close()
    conn.close()
except Exception as e:
    logging.error(f"Error loading data from database: {e}")
    raise

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Векторизация текстов
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Проверка размерностей данных
logging.info(f"X_train_vec shape: {X_train_vec.shape}")
logging.info(f"X_test_vec shape: {X_test_vec.shape}")

# Обучение и оценка моделей
models = [LogisticRegressionModel(), SVMModel(), RandomForestModel()]
results = []

for model in models:
    logging.info(f"Training and evaluating {model.__class__.__name__}")
    model.train(X_train_vec, y_train)
    evaluation_results = model.evaluate(X_test_vec, y_test)
    results.append({
        'Model': model.__class__.__name__,
        'Accuracy': evaluation_results['accuracy'],
        'Classification_Report': evaluation_results['classification_report']
    })

# Создание DataFrame для результатов
results_df = pd.DataFrame(results)

# Запись результатов в CSV файл
csv_file_path = 'evaluation_results.csv'
results_df.to_csv(csv_file_path, index=False)

# Уменьшение размерности данных для визуализации
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_vec.toarray())
X_test_pca = pca.transform(X_test_vec.toarray())

# Визуализация моделей на данных с уменьшенной размерностью
visualization_models = [
    VisualizationModel(LogisticRegression),
    VisualizationModel(SVC, kernel='linear'),
    VisualizationModel(RandomForestClassifier)
]

for vis_model in visualization_models:
    vis_model.train(X_train_pca, y_train)
    vis_model.visualize(X_train_pca, y_train)

# Ручной ввод примера для проверки модели
if __name__ == "__main__":
    text = input('Введите текст для проверки модели: ')
    try:
        preprocessor = TextPreprocessor(language='russian')
        preprocessed_text = preprocessor.preprocess_text(text)
        logging.info(f"Обработанный текст: {preprocessed_text}")

        vectorized_text = vectorizer.transform([preprocessed_text])
        for model in models:
            prediction = model.predict(vectorized_text)
            if prediction[0] == 0:
                text_prediction = 'Не молния'
            else:
                text_prediction = "Молния"
            logging.info(f"Предсказание модели {model.__class__.__name__}: {text_prediction}")
    except ValueError as e:
        logging.error(f"Error during text preprocessing: {e}")