import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from random_forest_package.exceptions import VisualizationError


class ModelVisualizer:
    def __init__(self, model):
        self.model = model

    def _extracted_from_plot_precision_recall_curve(self, arg0, arg1, arg2):
        plt.xlabel(arg0)
        plt.ylabel(arg1)
        plt.title(arg2)

    def plot_confusion_matrix(self, X, y, normalize=False):
        try:
            y_pred = self.model.predict(X)
            cm = confusion_matrix(y, y_pred, normalize='true' if normalize else None)
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
            self._extracted_from_plot_precision_recall_curve(
                'Predicted', 'True', 'Confusion Matrix'
            )
            plt.show()
        except Exception as e:
            raise VisualizationError(f"Error plotting confusion matrix: {e}") from e

    def plot_roc_curve(self, X, y):
        try:
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            self._extracted_from_plot_precision_recall_curve(
                'False Positive Rate',
                'True Positive Rate',
                'Receiver Operating Characteristic',
            )
            plt.legend(loc="lower right")
            plt.show()
        except Exception as e:
            raise VisualizationError(f"Error plotting ROC curve: {e}") from e

    def plot_precision_recall_curve(self, X, y):
        try:
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)

            plt.figure()
            plt.plot(recall, precision, color='b', lw=2)
            self._extracted_from_plot_precision_recall_curve(
                'Recall', 'Precision', 'Precision-Recall Curve'
            )
            plt.show()
        except Exception as e:
            raise VisualizationError(f"Error plotting precision-recall curve: {e}") from e
