from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay


class VisualizationModel:
    def __init__(self, model_class, **kwargs):
        self.model = model_class(**kwargs)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def visualize(self, X, y):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            self.model, X, ax=ax, grid_resolution=50, plot_method="contourf", alpha=0.8
        )
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", s=50)
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
        ax.add_artist(legend1)
        plt.title(f"{self.model.__class__.__name__} Decision Boundary")
        plt.show()

