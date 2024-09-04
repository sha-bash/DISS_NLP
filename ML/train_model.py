import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging

class ModelTrainer:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.vectorizer = TfidfVectorizer()
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'SVM': SVC(kernel='linear'),
            'RandomForest': RandomForestClassifier()
        }
        self.trained_models = {}

    def train_models(self):
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=42)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        for name, model in self.models.items():
            logging.info(f"Training {name} model")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = int(accuracy_score(y_test, y_pred) * 100)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.trained_models[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': report
            }
            logging.info(f"{name} Модель обучена с точностью: {accuracy}%")

    def get_trained_models(self):
        return self.trained_models, self.vectorizer