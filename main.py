import logging
import psycopg2
from utils.config import get_connection
from preprocessing.preprocessing import TextPreprocessor
from ML.train_model import ModelTrainer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
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

    # Проверка данных
    df = pd.DataFrame({'text': texts, 'label': labels})
    print(df['label'].value_counts())
    sns.countplot(x='label', data=df)
    plt.title('Распределение классов')
    plt.show()

    # Распределение длин текстов
    df['text_length'] = df['text'].apply(len)
    plt.figure(figsize=(8, 6))
    sns.histplot(df['text_length'], bins=50, kde=True)
    plt.title('Распределение длин текстов')
    plt.xlabel('Длина текста')
    plt.ylabel('Частота')
    plt.show()

    trainer = ModelTrainer(texts, labels)
    trainer.train_models()
    trained_models, vectorizer, scaler, pca = trainer.get_trained_models()

    # # Ручной ввод для тестирования
    # preprocessor = TextPreprocessor(language='russian')
    # text = input('Введите текст для проверки модели: ')
    # preprocessed_text = preprocessor.preprocess_text(text)
    # vectorized_text = vectorizer.transform([preprocessed_text])
    # scaled_text = scaler.transform(vectorized_text)
    # pca_text = pca.transform(scaled_text.toarray())

    # for name, info in trained_models.items():
    #     model = info['model']
    #     prediction = model.predict(pca_text)
    #     if prediction[0] == 0:
    #         text_prediction = 'Не молния'
    #     else:
    #         text_prediction = "Молния"
    #     logging.info(f"Предсказание модели {name}: {text_prediction}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()