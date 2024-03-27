import pandas as pd
import numpy as np
import pickle
from log import Log
from dataset import Dataset
from metrics import Metrics
from efc import EnergyBasedFlowClassifier
import time
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV 
import matplotlib.pyplot as plt

ENV = "/home/alvaro/tcc/experiments"



class Classification(Dataset, Metrics, Log):
    def __init__(self, instructions):
        super().__init__()
        self.log("Classification initialized")
        if instructions['Grid'] is not None:
            self.get_tun_clf(
                instructions['Dataset'], 
                instructions['Pretrained'], 
                instructions['Save'], 
                instructions['Grid']
            )
        else:
            self.get_std_clf(
                instructions['Dataset'], 
                instructions['Pretrained'], 
                instructions['Save']
            )
        super().__del__()  
    def __del__(self):
        self.log("Classification finished")
 
    
    def load_pretrained(self, dataset):
        self.log("Loading the model...")
        with open(f'{ENV}/models/efc_{dataset}.pkl', 'rb') as f:
            clf = pickle.load(f)

        self.log("Loading the test set...")
        testdf = pd.read_csv(f'{ENV}/data/efc_{dataset}_test_set.csv')
        y_test = testdf[' Label']
        X_test = testdf.drop(columns=[' Label', 'Unnamed: 0'])
        self.log(f"Showing dataset:\n{y_test.head()}")
        return clf, X_test, y_test
    
    def train_model(self, dataset, save):
        #getting dataset
        X_train, X_test, y_train, y_test = self.get_split_df(dataset, 0.3)
        self.log("Model defined")
        clf = EnergyBasedFlowClassifier(n_bins=10, cutoff_quantile=0.95)
        #training 
        self.log("Training the model...")
        self.func_time("Model Training",clf.fit(X_train, y_train, base_class=0))
        if save == True:
            self.log("Saving the model and test set...")
            start_time = time.time()
            with open(f'{ENV}/models/efc_{dataset}.pkl', 'wb') as f:
                pickle.dump(clf, f)
            testdf = pd.concat([X_test, y_test], axis = 1)
            testdf.to_csv(f'{ENV}/data/efc_{dataset}_test_set.csv')
            end_time = time.time()
            total_time = end_time - start_time
            self.log(f"Model and test set saved in {total_time // 60} minutes {int(total_time % 60)} seconds")
        return clf, X_test, y_test
    
    def classification(self, clf, X_test, y_test):
        #classification
        self.log("Executing the classification...")
        start_time = time.time()
        y_pred, y_energies = clf.predict(X_test, return_energies=True)
        end_time = time.time()
        total_time = end_time - start_time
        self.log("Classification executed")
        self.log(f"Classification time: {total_time // 60} minutes {int(total_time % 60)} seconds")
        return clf, y_test, y_pred, y_energies
    
    def plot_metrics(self, clf, y_test, y_pred, y_energies):     
        #including metrics
        self.log("Plotting the energies for EFC...")
        self.plot_efc_energies(y_test, y_energies, clf)
        self.log("Plotting the ROC curve...")
        self.plot_roc_curve(y_test, y_pred)
        self.log("Plotting the confusion matrix...")
        self.plot_confusion_matrix(y_test, y_pred)
        self.log("Saving the classification report...")
        self.save_classification_report(y_test, y_pred)  

    def get_std_clf(self, dataset, pretrained = False, save = False):
        
        #defining model
        if pretrained == True:
            clf, X_test, y_test = self.load_pretrained(dataset)
        else:
            clf, X_test, y_test = self.train_model(self, dataset, save)
                
        self.plot_metrics(self.classification(clf, X_test, y_test))

    def get_tun_clf(self, dataset, pretrained=False, save=False, param_grid=None):
        
        X_train, X_test, y_train, y_test = self.get_split_df(dataset, 0.3)
        self.log("Splitting the dataset...")
        # unify X train and test and y train and test
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])

        self.log("Model defined")
        clf = EnergyBasedFlowClassifier(n_jobs=-1)

        self.log("Tuning the classifier...")
        search = HalvingGridSearchCV(clf, param_grid, scoring='f1', cv=5, verbose=5)
        search.fit(X, y, base_class=0)
        
        best_clf = search.best_estimator_
        best_params = search.best_params_
        self.log(f"Best parameters: {best_params}")
        
        results = pd.DataFrame(search.cv_results_)
        results["params_str"] = results.params.apply(str)
        results.drop_duplicates(subset=("params_str", "iter"), inplace=True)
        mean_scores = results.pivot(
            index="iter", columns="params_str", values="mean_test_score"
        )
        ax = mean_scores.plot(legend=False, alpha=0.6)

        labels = [
            f"iter={i}\nn_samples={search.n_resources_[i]}\nn_candidates={search.n_candidates_[i]}"
            for i in range(search.n_iterations_)
        ]

        ax.set_xticks(range(search.n_iterations_))
        ax.set_xticklabels(labels, rotation=45, multialignment="left")
        ax.set_title("Scores of candidates over iterations")
        ax.set_ylabel("mean test score", fontsize=15)
        ax.set_xlabel("iterations", fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{ENV}/imgs/parameter_tuning.png')
    
      


efc_param_grid = {
    "n_bins": [30],
    "cutoff_quantile": [0.95]
}
instructions= {
    "Dataset": "CICIDS2017",
    "Pretrained": False,
    "Model": "EFC",
    "Save": False,
    "Grid": efc_param_grid
}
clf = Classification(instructions)
del clf
