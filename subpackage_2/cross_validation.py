import subpackage_1.KNN_data_collection as KNN_module
import subpackage_1.generate_predictions as gp
from model_metrics import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CvKNN(KNN_module.KNN):
    def __init__(self,model_type,k=3,num_folds=5):
        KNN_module.KNN.__init__(self,model_type,k)
        self.num_folds=num_folds
        print('Number of CV folds: '+str(num_folds))

    def perform_cv(self,k_values):
        x_folds=np.array_split(self.x_train,self.num_folds)
        y_folds=np.array_split(self.y_train,self.num_folds)
        self.__k_results=[]
        self.__k_values=k_values
        if self.model_type=='regressor':
            for k in k_values:
                fold_results=[]
                for i in range(self.num_folds):
                    fold_pred=gp.generate_predictions(self,x_folds[i],'train')
                    fold_true=y_folds[i]
                    fold_results.append(model_rmse(fold_true,fold_pred))
                self.__k_results.append(np.mean(fold_results))
        elif self.model_type=='classifier':
            for k in k_values:
                fold_results=[]
                for i in range(self.num_folds):
                    fold_pred=gp.generate_predictions(self,x_folds[i],'train')
                    fold_true=y_folds[i]
                    fold_results.append(model_midclassification(fold_true,fold_pred))
                self.__k_results.append(np.mean(fold_results))
    def get_cv_results(self):
        for i in range(len(self.__k_values)):
            print('k='+str(self.__k_values[i])+': '+str(self.__k_results[i]))
        sns.lineplot(y=self.__k_results,x=self.__k_values)
        plt.show()

    def get_best_k(self):
        min_position=np.argmin(self.__k_results)
        best_k=self.__k_values[min_position]
        self.best_k=best_k
        print('Best k: '+str(best_k))
        print('CV result: '+str(self.__k_results[min_position]))
        return best_k
