import logging
import psycopg2
from utils.config import get_connection
from preprocessing.preprocessing import TextPreprocessor
from ML.train_model import ModelTrainer

def main():
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

    trainer = ModelTrainer(texts, labels)
    trainer.train_models()
    trained_models, vectorizer = trainer.get_trained_models()

    # Ручной ввод для тестирования
    preprocessor = TextPreprocessor(language='russian')
    text = input('Введите текст для проверки модели: ')
    preprocessed_text = preprocessor.preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])

    for name, info in trained_models.items():
        model = info['model']
        prediction = model.predict(vectorized_text)
        if prediction[0] == 0:
            text_prediction = 'Не молния'
        else:
            text_prediction = "Молния"
        logging.info(f"Предсказание модели {name}: {text_prediction}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()