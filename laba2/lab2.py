from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error,root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge


df = pd.read_csv("D:/lab_AI/laba1/processed_titanic.csv")


X = df.drop('Age', axis=1)
y = df['Age']
#n=2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

liner_reg_model = LinearRegression()
liner_reg_model.fit(X_train, y_train)
y_pred_test_liner = liner_reg_model.predict(X_test)
#ostatok = y_test - y_pred_test_liner

# poly_features = PolynomialFeatures(n)
# X_train_poly = poly_features.fit_transform(X_train)
# X_test_poly = poly_features.transform(X_test)
# poly_model = LinearRegression()
# poly_model.fit(X_train_poly, y_train)
# y_pred_test_poly = poly_model.predict(X_test_poly)
# residuals_poly = y_test - y_pred_test_poly

MSE = mean_squared_error(y_test, y_pred_test_liner)
RMSE = root_mean_squared_error(y_test, y_pred_test_liner)
MAE = mean_absolute_error(y_test, y_pred_test_liner)

# MSE_pol = mean_squared_error(y_test, y_pred_test_poly)
# RMSE_pol = root_mean_squared_error(y_test, y_pred_test_poly)
# MAE_pol = mean_absolute_error(y_test, y_pred_test_poly)

print('MSE:',MSE)
print('RMSE:',RMSE)
print('MAE:',MAE)

# print('MSE:',MSE_pol)
# print('RMSE:',RMSE_pol)
# print('MAE:',MAE_pol)

res = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test_liner,
})

l2_linear_model =Ridge(alpha=2.0)
l2_linear_model.fit(X_train, y_train)
y_pred_test_l2 = l2_linear_model.predict(X_test)

MSE_L2 = mean_squared_error(y_test, y_pred_test_l2)
RMSE_L2 = root_mean_squared_error(y_test, y_pred_test_l2)
MAE_L2 = mean_absolute_error(y_test, y_pred_test_l2)

print('MSE_L2:',MSE_L2)
print('RMSE_L2:',RMSE_L2)
print('MAE_L2:',MAE_L2)

#plt.scatter(y_pred_test_liner, residuals)
# plt.scatter(y_pred_test_poly, residuals_poly)

df['Transported'] = df['Transported'].astype(int) 
X = df.drop('Transported', axis=1)
y= df['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
y_pred_test_log = log_reg_model.predict(X_test)


print("accuracy:",accuracy_score(y_test, y_pred_test_log))
print("precision:",precision_score(y_test, y_pred_test_log))
print("recall:",recall_score(y_test, y_pred_test_log))
print("f1:",f1_score(y_test, y_pred_test_log))

###дисбаланс классов

cm_log = confusion_matrix(y_test, y_pred_test_log)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print(df.value_counts('Transported'))
