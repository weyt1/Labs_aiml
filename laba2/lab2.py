from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error,root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures



df = pd.read_csv("D:/lab_AI/laba1/processed_titanic.csv")

print(df.head())
df.info()


X = df.drop('Age', axis=1)
y = df['Age']

df.info()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

liner_reg_model = LinearRegression()
liner_reg_model.fit(X_train, y_train)
y_pred_test_liner = liner_reg_model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_test_liner)
RMSE = root_mean_squared_error(y_test, y_pred_test_liner)
MAE = mean_absolute_error(y_test, y_pred_test_liner)

print('MSE:',MSE)
print('RMSE:',RMSE)
print('MAE:',MAE)

res = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred_test_liner,
})

print(res.head())

df['VIP'] = df['VIP'].astype(int) 
X = df.drop('VIP', axis=1)
y= df['VIP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
y_pred_test_log = log_reg_model.predict(X_test)


print("accuracy:",accuracy_score(y_test, y_pred_test_log))
print("precision:",precision_score(y_test, y_pred_test_log))
print("recall:",recall_score(y_test, y_pred_test_log))
print("f1:",f1_score(y_test, y_pred_test_log))

cm_log = confusion_matrix(y_test, y_pred_test_log)
plt.figure(figsize=(4, 3))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print(df.value_counts('VIP'))
