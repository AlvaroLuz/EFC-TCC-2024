import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from log import Log
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, classification_report

class Metrics(Log):
    def plot_efc_energies(self, y_test, y_energies, clf):
        # ploting energies
        benign = np.where(y_test == 0)[0]
        malicious = np.where(y_test == 1)[0]

        benign_energies = y_energies[benign]
        malicious_energies = y_energies[malicious]
        cutoff = clf.estimators_[0].cutoff_

        bins = np.histogram(y_energies, bins=60)[1]

        plt.hist(
            malicious_energies,
            bins,
            facecolor="#006680",
            alpha=0.7,
            ec="white",
            linewidth=0.3,
            label="malicious",
        )
        plt.hist(
            benign_energies,
            bins,
            facecolor="#b3b3b3",
            alpha=0.7,
            ec="white",
            linewidth=0.3,
            label="benign",
        )
        plt.axvline(cutoff, color="r", linestyle="dashed", linewidth=1)
        plt.legend()

        plt.xlabel("Energy", fontsize=12)
        plt.ylabel("Density", fontsize=12)

        #plt.show()
        plt.savefig('../imgs/energies.png')

    def plot_roc_curve(self, y_test, y_pred):

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig('../imgs/roc_curve.png')

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        cm.plot()
        plt.savefig('../imgs/confusion_matrix.png')
        #plt.show()

    def save_classification_report(self, y_test, y_pred):
        target_names = ["Benign", "Malicious"]
        report = classification_report(y_test, y_pred, target_names=target_names)
        self.log(f"Showing Classification Report:\n{report}")
        with open('../imgs/classification_report.txt', 'w') as file:
            file.write(report)
