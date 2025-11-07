import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, roc_curve, auc

df = pd.read_csv("D:/lab_AI/laba1/processed_titanic.csv")

X = df.drop('Age', axis=1)
y = df['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

dt_regressor_model = DecisionTreeRegressor()
dt_regressor_model.fit(X_train, y_train)
y_pred_test = dt_regressor_model.predict(X_test)

#plot_tree(dt_regressor_model, proportion=True)

# x_range = np.linspace(y_test.min(), y_test.max(), 100)
# plt.scatter(y_test,y_pred_test)
# plt.plot(x_range, x_range)
# plt.xlabel('y_test')
# plt.ylabel('y_pred_test')

MAE = mean_absolute_error(y_test,y_pred_test)
MSE = mean_squared_error(y_test,y_pred_test)
RMSE = root_mean_squared_error(y_test,y_pred_test)

print("MAE",MAE)
print("MSE",MSE)
print('RMSE',RMSE)

df['Transported'] = df['Transported'].astype(int) 
X = df.drop('Transported', axis=1)
y= df['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

dt_classifier_model = DecisionTreeClassifier()
dt_classifier_model.fit(X_train, y_train)
y_proba = dt_classifier_model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])

plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
auc_metric = auc(fpr, tpr)
print("auc_metric: ",auc_metric)
plt.show()
