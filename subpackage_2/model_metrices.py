import subpackage_1.KNN_data_collection as knn
import subpackage_1.generate_predictions as gp
import numpy as np

def model_accuracy(knn_type, actual, predicted):
    if knn_type == 'regressor':
        result = predicted/actual * 100
    elif knn_type == 'classifier':
        if (actual == predicted):
            result = 1
        else:
            result = 0
    return result
    
    def model_rmse(knn_type, actual, predicted):
    total_size = np.size(actual)
    rmse = 0
    if knn_type == 'regressor':
        rmse = np.sqrt(np.mean(predicted - actual)**2)
    elif knn_type == 'classifer':
        rmse = (total_size - np.sum((actual - predicted) != 0))/total_size
    return rmse
    
    def model_mape(knn_type, actual, predicted):
    total_size = np.size(actual)
    mape = 0
    if knn_type == 'regressor':
        mape = (np.sum(np.abs(actual - predicted)/actual)/total_size)* 100
    elif knn_type == 'classifer':
        mape = ((total_size - np.sum((actual - predicted) != 0))/total_size)* 100
    return mape

 
def assesment_metrics(knn_model, train_obs, test_obs):
    classification_model=knn.KNN('classifier',2)
    classification_model.load_csv('Iris.csv','Species')
    classification_model.train_test_split(0.4)
    
    regression_model=knn.KNN('regressor')
    regression_model.load_csv('AutoMPG.csv','mpg')
    regression_model.train_test_split(0.33)
    
    if knn_model == 'regressor':
        train_result = gp.generate_prediction(regression_model, train_obs, 'train').astype(float)
        test_result = gp.generate_prediction(regression_model, test_obs, 'test').astype(float)
    elif knn_model == 'classifier':
        train_result = gp.generate_prediction(classification_model, train_obs, 'train')
        test_result = gp.generate_prediction(classification_model, test_obs, 'test')
   
    t_accuracy = model_accuracy(knn_model, test_result,train_result)
    test_rmse = model_rmse(knn_model,test_result, train_result)
    test_mape = model_mape(knn_model,test_result,train_result  )
    #print(t_accuracy, test_rmse)
    return t_accuracy, test_rmse, test_mape
    