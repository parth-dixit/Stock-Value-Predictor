import pandas as pd
from sklearn.preprocessing import StandardScaler


#Reading the data
mydata = pd.read_csv(r'C:\Users\Parth Dixit\Downloads\TIRUMALCHM.NS.csv',index_col = 'Date')

#Retrieving the appropriate data from mydata
start_train = '2018-01-01'
end_train = '2019-06-2-'
data_train = mydata.ix[start_train:end_train]
X_columns = list(mydata.drop(['Close'],axis=1).columns)
y_columns = 'Close'
X_train = data_train[X_columns]
y_train = data_train[y_columns]
X_train.shape

start_test = '2019-06-21'
end_test = '2019-09-03'
data_test = mydata.ix[start_test:end_test]
X_columns = list(mydata.drop(['Close'],axis=1).columns)
y_columns = 'Close'
X_test = data_test[X_columns]
y_test = data_test[y_columns]
X_test.shape

#Scaling the data using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

param_grid = {
    
    "alpha" : [1e-5,3e-5,1e-4],
    "eta0" : [0.01,0.03,0.1],
}


#Predicting value using linear Regression
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

lr = linear_model.SGDRegressor(penalty='l2',max_iter=1000)
grid_search = GridSearchCV(lr,param_grid,cv=5,scoring = 'neg_mean_absolute_error')
grid_search.fit(X_scaled_train,y_train)

print(grid_search.best_params_)

lr_best = grid_search.best_estimator_
predictions_lr = lr_best.predict(X_scaled_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

print('MSE : {0:.3f}'.format(mean_squared_error(y_test,predictions)))
print('MAE : {0:.3f}'.format(mean_absolute_error(y_test,predictions)))
print('R2 : {0:.3f}'.format(r2_score(y_test,predictions)))


#Predicting value using Random Forest Regression
param_grid = {
    "max_depth" : [30,50],
    "min_samples_split" : [5,10,20],
}

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000)

grid_search = GridSearchCV(rf,param_grid,cv=5,scoring = 'neg_mean_absolute_error')
grid_search.fit(X_train,y_train)

print(grid_search.best_params_)
rf_best = grid_search.best_estimator_
predictions_rf = rf_best.predict(X_test)

print('MSE : {0:.3f}'.format(mean_squared_error(y_test,predictions)))
print('MAE : {0:.3f}'.format(mean_absolute_error(y_test,predictions)))
print('R2 : {0:.3f}'.format(r2_score(y_test,predictions)))

#Predicting value using Support Vector machine using linear kernel
param_grid = {
    "C" : [1000,3000,10000],
    "epsilon" : [0.00001,0.00003,0.0001],
}

from sklearn.svm import SVR

svr = SVR(kernel = 'linear')

grid_search = GridSearchCV(svr,param_grid,cv=5,scoring = 'neg_mean_absolute_error')
grid_search.fit(X_scaled_train,y_train)

print(grid_search.best_params_)
svr_best = grid_search.best_estimator_
predictions_svr = svr_best.predict(X_scaled_test)

print('MSE : {0:.3f}'.format(mean_squared_error(y_test,predictions)))
print('MAE : {0:.3f}'.format(mean_absolute_error(y_test,predictions)))
print('R2 : {0:.3f}'.format(r2_score(y_test,predictions)))


#Plotting the predicted value got from all the methods vs the actual value
import matplotlib.pyplot as plt

dates = data_test.index.values
plot_truth, = plt.plot(dates,y_test,'k')
plot_lr, = plt.plot(dates,predictions_lr,'r')
plot_rf, = plt.plot(dates,predictions_rf,'b')
plot_svr, = plt.plot(dates,predictions_svr,'g')
plt.legend([plot_truth,plot_lr,plot_rf,plot_svr],['Truth','Linear Regression','Random Forest','SVR'])
plt.title('Stock Predictions vs Actual Price')
plt.show()