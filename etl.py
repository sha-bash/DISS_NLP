import pandas as pd
from preprocessing.preprocessing import TextPreprocessor, TextVectorizer
import psycopg2
from psycopg2 import sql
from utils.config import read_json, get_connection
import numpy as np

class ETLProcessor:
    def __init__(self, csv_file_path, db_config):
        self.csv_file_path = csv_file_path
        self.db_config = db_config
        self.preprocessor = TextPreprocessor(language='russian')
        self.vectorizer = TextVectorizer()

    def extract(self):
        # Чтение данных из CSV
        self.data = pd.read_csv(self.csv_file_path, sep=';', on_bad_lines='warn', encoding='utf-8')
        if 'Текст' not in self.data.columns or 'Метка' not in self.data.columns:
            raise ValueError("CSV file must contain 'Текст' and 'Метка' columns")
        return self.data

    def transform(self):
        # Предобработка текстов
        self.data['processed_text'] = self.data['Текст'].apply(self.preprocessor.preprocess_text)
        
        # Векторизация текстов
        text_list = self.data['processed_text'].tolist()
        X_vec, self.vectorizer = self.vectorizer.vectorize_text(text_list)
        
        # Добавление векторизованных данных в DataFrame
        self.data['vectorized_text'] = list(X_vec.toarray())
        
        # Добавление столбца label_id
        self.data['label_id'] = self.data['Метка'].apply(lambda x: 1 if x == 'Молния' else 0)
        
        return self.data

    def load(self):
        # Подключение к базе данных
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Создание таблицы, если она не существует
        create_table_query = """
        CREATE TABLE IF NOT EXISTS processed_texts (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            processed_text TEXT NOT NULL,
            vectorized_text REAL[] NOT NULL,
            label TEXT NOT NULL,
            label_id INT NOT NULL
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        
        # Вставка данных в таблицу
        for index, row in self.data.iterrows():
            # Преобразование numpy.ndarray в список
            vectorized_text_list = row['vectorized_text'].tolist()
            insert_query = sql.SQL("""
            INSERT INTO processed_texts (text, processed_text, vectorized_text, label, label_id)
            VALUES (%s, %s, %s, %s, %s)
            """)
            cursor.execute(insert_query, (row['Текст'], row['processed_text'], vectorized_text_list, row['Метка'], row['label_id']))
        
        conn.commit()
        cursor.close()
        conn.close()

if __name__ == "__main__":
    csv_file_path = 'private/Data.csv'
    conn = get_connection()
    
    # Убедитесь, что conn содержит правильные параметры для подключения к базе данных
    db_config = {
        'dbname': conn.info.dbname,
        'user': conn.info.user,
        'password': conn.info.password,
        'host': conn.info.host,
        'port': conn.info.port
    }
    
    etl_processor = ETLProcessor(csv_file_path, db_config)
    etl_processor.extract()
    etl_processor.transform()
    etl_processor.load()