class KNN: 
    def __init__(self, features, dummy_labels, k =3, model_type):
        self.features = np.array(features)
        self.labels = np.array(dummy_labels)
        self.k = k
        self.model_type = model_type
        
    def load_csv_file(path):
        data = []
        with open(path ,'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                data.append(row)
        return np.array(data)
    
    def train_test_split(dataset, test_size = 0.3):
                            dataset_ = np.array(dataset)
                            np.randonm.shuffle(dataset_) # to avoid bias
                            threshold = int(dataset_.shape[0] * 0.1)
                            x_test = dataset_[:threshold, :-1]
                            y_test = dataset_[:threshold, -1]
                            x_train = dataset_[threshold:, :-1]
                            y_train = dataset_[threshold:, -1]
        return x_test, y_test, x_train, y_train