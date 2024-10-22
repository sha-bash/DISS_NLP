import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler(with_mean=False)  # with_mean=False для разреженных матриц
        self.pca = PCA(n_components=0.95)  # Сохраняем 95% дисперсии
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'SVM': SVC(kernel='poly', degree=2, probability=True),
            'RandomForest': RandomForestClassifier()
        }
        self.trained_models = {}

    def train_models(self):
        X_train, X_test, y_train, y_test = train_test_split(self.texts, self.labels, test_size=0.2, random_state=42)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Нормализация данных
        X_train_scaled = self.scaler.fit_transform(X_train_vec)
        X_test_scaled = self.scaler.transform(X_test_vec)

        # Применение PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled.toarray())  # Преобразуем в плотную матрицу для PCA
        X_test_pca = self.pca.transform(X_test_scaled.toarray())

        # График распределения данных после PCA
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis')
        plt.title('PCA распределение')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

        for name, model in self.models.items():
            logging.info(f"Training {name} model")
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
            y_pred_proba = model.predict_proba(X_test_pca)[:, 1] if hasattr(model, "predict_proba") else None
            accuracy = int(accuracy_score(y_test, y_pred) * 100)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.trained_models[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix,
                'report': report,
                'y_pred_proba': y_pred_proba
            }
            logging.info(f"{name} Модель обучена с точностью(accuracy): {accuracy}%, точность(precision): {precision}, отзыв(recall): {recall}, F1: {f1}")

            # Построение матрицы ошибок
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Матрица ошибок для {name}')
            plt.xlabel('Предсказание')
            plt.ylabel('Факт')
            plt.show()

            # Построение ROC-кривой
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Ложноположительный результат')
                plt.ylabel('Истинно положительный результат')
                plt.title(f'ROC кривая {name}')
                plt.legend(loc="lower right")
                plt.show()

            # Построение Precision-Recall кривой
            if y_pred_proba is not None:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                plt.figure(figsize=(6, 4))
                plt.plot(recall, precision, label=f'Precision-Recall curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall кривая {name}')
                plt.legend(loc="lower left")
                plt.show()

            # Распределение вероятностей предсказаний
            if y_pred_proba is not None:
                plt.figure(figsize=(6, 4))
                sns.histplot(y_pred_proba, bins=50, kde=True)
                plt.title(f'Распределение вероятностей предсказаний {name}')
                plt.xlabel('Вероятность')
                plt.ylabel('Частота')
                plt.show()

        # График важности признаков для RandomForest
        if 'RandomForest' in self.trained_models:
            rf_model = self.trained_models['RandomForest']['model']
            importances = rf_model.feature_importances_
            indices = importances.argsort()[::-1]
            plt.figure(figsize=(10, 6))
            plt.title("Важные признаки")
            plt.bar(range(X_train_pca.shape[1]), importances[indices], align="center")
            plt.xticks(range(X_train_pca.shape[1]), indices)
            plt.xlabel('Индекс характеристики')
            plt.ylabel('Важность')
            plt.show()

    def get_trained_models(self):
        return self.trained_models, self.vectorizer, self.scaler, self.pca