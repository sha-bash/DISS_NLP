import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
            return {
                'accuracy': accuracy,
                'classification_report': classification_report_dict
            }
        except Exception as e:
            logging.error(f"Error evaluating LogisticRegressionModel: {e}")
            return {
                'accuracy': None,
                'classification_report': None
            }

    def visualize(self, X, y):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            self.model, X, ax=ax, grid_resolution=50, plot_method="contourf", alpha=0.8
        )
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", s=50)
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax.add_artist(legend1)
        plt.title("Logistic Regression Decision Boundary")
        plt.show()

class SVMModel:
    def __init__(self):
        self.model = SVC(kernel='linear')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
            return {
                'accuracy': accuracy,
                'classification_report': classification_report_dict
            }
        except Exception as e:
            logging.error(f"Error evaluating SVMModel: {e}")
            return {
                'accuracy': None,
                'classification_report': None
            }

    def visualize(self, X, y):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        # Создаем копию модели без проверки количества признаков
        model_clone = clone(self.model)
        model_clone.fit(X, y)
        
        DecisionBoundaryDisplay.from_estimator(
            model_clone, X, ax=ax, grid_resolution=50, plot_method="contourf", alpha=0.8
        )
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", s=50)
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax.add_artist(legend1)
        plt.title("SVM Decision Boundary")
        plt.show()

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
            return {
                'accuracy': accuracy,
                'classification_report': classification_report_dict
            }
        except Exception as e:
            logging.error(f"Error evaluating RandomForestModel: {e}")
            return {
                'accuracy': None,
                'classification_report': None
            }

    def visualize(self, X, y):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            self.model, X, ax=ax, grid_resolution=50, plot_method="contourf", alpha=0.8
        )
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", s=50)
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax.add_artist(legend1)
        plt.title("Random Forest Decision Boundary")
        plt.show()


