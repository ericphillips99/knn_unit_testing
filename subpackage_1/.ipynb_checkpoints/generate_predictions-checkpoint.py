import numpy as np
import csv
import warnings

def euclidean_distance(point1,point2):
    distance=0
    for i in range(len(point1)):
        distance+=(point1[i]-point2[i])**2
    return np.sqrt(distance)

def generate_prediction(knn_model,new_obs,subset): # Fix DRY violation
    if subset not in ['train','test','all']:
        raise ValueError(str(subset)+' is invalid for subset. Must be one of "train", "test", or "all".')
    # Compute distance from new observation to every sample in training set
    if subset=='train':
        distances=np.apply_along_axis(lambda x:euclidean_distance(x,new_obs), 1, knn_model.x_train)
        # Get index positions of k closest points
        k_indices=np.argsort(distances)[:knn_model.k]
        if knn_model.model_type=='regressor':
            return np.mean(knn_model.y_train[k_indices])
        elif knn_model.model_type=='classifier':
            classes,counts=np.unique(knn_model.y_train[k_indices],return_counts=True)
            # Need to test if warning works as expected
            if len(counts)>1:
                top_counts=np.sort(counts)[-2:]
                if top_counts[0]==top_counts[1]:
                    warnings.warn('Warning: A tie has occurred (top two classes in K nearest neighbors have the same number of occurances). Classification depends on the order of the training data.')
            return classes[np.argmax(counts)]
        else:
            print('Invalid model type for inputted model')
            return None
    elif subset=='test':
        distances=np.apply_along_axis(lambda x:euclidean_distance(x,new_obs), 1, knn_model.x_test)
        # Get index positions of k closest points
        k_indices=np.argsort(distances)[:knn_model.k]
        if knn_model.model_type=='regressor':
            return np.mean(knn_model.y_test[k_indices])
        elif knn_model.model_type=='classifier':
            classes,counts=np.unique(knn_model.y_test[k_indices],return_counts=True)
            # Need to test if warning works as expected
            if len(counts)>1:
                top_counts=np.sort(counts)[-2:]
                if top_counts[0]==top_counts[1]:
                    warnings.warn('Warning: A tie has occurred (top two classes in K nearest neighbors have the same number of occurances). Classification depends on the order of the training data.')
            return classes[np.argmax(counts)]
        else:
            print('Invalid model type for inputted model')
            return None
    else:
        distances=np.apply_along_axis(lambda x:euclidean_distance(x,new_obs), 1, knn_model.x)
        # Get index positions of k closest points
        k_indices=np.argsort(distances)[:knn_model.k]
        if knn_model.model_type=='regressor':
            return np.mean(knn_model.y[k_indices])
        elif knn_model.model_type=='classifier':
            classes,counts=np.unique(knn_model.y[k_indices],return_counts=True)
            # Need to test if warning works as expected
            if len(counts)>1:
                top_counts=np.sort(counts)[-2:]
                if top_counts[0]==top_counts[1]:
                    warnings.warn('Warning: A tie has occurred (top two classes in K nearest neighbors have the same number of occurances). Classification depends on the order of the training data.')
            return classes[np.argmax(counts)]
        else:
            print('Invalid model type for inputted model')
            return None
