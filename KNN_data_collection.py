import numpy as np
import csv
class KNN:
    data = []
    y= []
    x=[]
    def __init__(self,model_type,k=3):
        self.k = k
        if model_type=='regressor' or model_type=='classifier':
            self.model_type = model_type
        else:
            raise ValueError('Model type must be "regressor" or "classifier"')

    def load_csv_file(self,path,response = 'Species'):
        with open(path ,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                KNN.data.append(row)
        
        if response in data[0]:
            for row in data:
                response_idx = np.where(data[0] == response)
                KNN.y.append(row[response_idx])
                #del row[KNN.y]
                KNN.x.append(row[~KNN.y]) # I would need your help on this

        self.x=np.array(x) # Predictor variables [store as instance attribute instead of returning]
        self.y=np.array(y) # Response variables [store as instance attribute instead of returning]
        print('Dataset successfully loaded.')

    def train_test_split(self, test_size = 0.3):
        # Need to select random rows from both x and y: possible solution could be to sample indexers from 0 to length of dataset without replacement using np.random.choice?
        # Could select subset of x and y based on these random index vals (e.g., these vals become the training set, rest become test, etc)
       
        # predictor variable
        predictor_ = np.array(x)
        threshold_p = int(predictor_.shape[0])
        random_indices_p  = np.random.choice(threshold_p, size = self.test_size*threshold_p, replace = False)
        self.y_test = predictor_[random_indices_p]
        self.y_train = [element for element in y_test if element not in random_indices_p]

        #reponse variable split
        response_ = np.array(y)
        threshold_r = int(response_.shape[0])
        random_indices_r  = np.random.choice(threshold_r, size = self.test_size*threshold_r, replace = False)
        self.x_test = predictor_[random_indices_r]
        self.x_train = [element for element in x_test if element not in random_indices_r]
        print('Train - test - split successfull') 