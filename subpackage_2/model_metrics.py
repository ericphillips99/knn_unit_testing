import subpackage_1.KNN_data_collection as knn
import subpackage_1.generate_predictions as gp
import numpy as np
def model_accuracy(actual, predicted): # for classifer
    """Model accuracy for Classification"""
    if np.mean(np.array(predicted)) == np.mean(np.array(actual)):
        result = 1
    else:
        result = 0
    print("Prediction Output: (1: Success, 0:Incorrect prediction)", result) 

def model_misclassification(actual, predicted): #for classifer
        misclassification_rate = 0
        total_size = np.size(np.array(actual))
        missclassify = np.mean(np.array(predicted)) != np.mean(np.array(actual))
        misclassification_rate = (total_size - missclassify)/total_size
        print('Misclassification Rate:',misclassification_rate)
        
            
def model_rmse(actual, predicted): # for regressor
    total_size = np.size(actual)
    rmse = 0
    rmse = np.sqrt(np.mean(predicted - actual)**2)
    return rmse
    
def model_mape(actual, predicted): # for regressor
    total_size = np.size(actual)
    mape = 0
    if knn_type == 'regressor':
        mape = (np.sum(np.abs(actual - predicted)/actual)/total_size)* 100
    return mape
 
def assesment_metrics(knn_type, train_obs, test_obs):
    test_accuracy = 0
    test_missclassification =0
    test_mape = 0
    test_rmse = 0
    
    if knn_type == 'regressor':
        train_result = gp.generate_predictions(regression_model, [train_obs, test_obs], 'train').astype(float)
        test_result = gp.generate_predictions(regression_model, [train_obs, test_obs], 'all').astype(float)
        test_rmse = model_rmse(test_result, train_result)
        test_mape = model_mape(test_result,train_result)
        
    elif knn_type == 'classifier':
        train_result = gp.generate_predictions(classification_model, [train_obs, test_obs], 'train')
        test_result = gp.generate_predictions(classification_model, [train_obs, test_obs], 'all')
        
        test_accuracy = model_accuracy(test_result,train_result)
        test_missclassification = model_misclassification(test_result,train_result)
   
    print('Accuracy of the Classifier model :', test_accuracy )
    print('Misclassification of Classifier model : ',test_missclassification)    
    print('RSME of regressor :', test_rmse)
    print('MAPE of regressor : ',test_mape)