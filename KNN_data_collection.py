import numpy as np
class KNN:
    def __init__(self,model_type,k=3):
        self.k = k
        if model_type=='regressor' or model_type=='classifier':
            self.model_type = model_type
        else:
            raise ValueError('Model type must be "regressor" or "classifier"')

    def load_csv_file(self,path,response):
        data = []
        with open(path ,'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                data.append(row)
        # Need to separate response variable from predictors (rest of the columns)
        # Need to remove column names after this is done (remove first row)
        self.x=np.array() # Predictor variables [store as instance attribute instead of returning]
        self.y=np.array() # Response variables [store as instance attribute instead of returning]
        print('Dataset successfully loaded.')

    def train_test_split(self, test_size = 0.3):
        # Need to select random rows from both x and y: possible solution could be to sample indexers from 0 to length of dataset without replacement using np.random.choice?
            # Could select subset of x and y based on these random index vals (e.g., these vals become the training set, rest become test, etc)
        dataset_ = np.array(dataset)
        np.randonm.shuffle(dataset_)
        threshold = int(dataset_.shape[0] * 0.1)
        x_test = dataset_[:threshold, :-1]
        y_test = dataset_[:threshold, -1]
        x_train = dataset_[threshold:, :-1]
        y_train = dataset_[threshold:, -1]
        return x_test, y_test, x_train, y_train # Store as self.x_test, self.y_test, etc. instead of returning
