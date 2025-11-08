import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt
df = pd.read_csv("D:/lab_AI/laba1/processed_titanic.csv")
def roc_curve_show(model_name,fpr,tpr):
    plt.plot(fpr, tpr, marker='o')
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.title(f"ROC curve: {model_name}")
    plt.show()
def accuracy_precision_recall(name_model,y_test, y_pred_test):
    print("#"*50)
    print(f"accuracy:{name_model}",accuracy_score(y_test, y_pred_test))
    print(f"precision:{name_model}",precision_score(y_test, y_pred_test))
    print(f"recall:{name_model}",recall_score(y_test, y_pred_test))

df['Transported'] = df['Transported'].astype(int) 
X = df.drop('Transported', axis=1)
y= df['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

forest_model = RandomForestClassifier(oob_score=True)
forest_model.fit(X_train,y_train)
y_pred_test_forest = forest_model.predict(X_test)
print("OOB Accuracy:", forest_model.oob_score_)
print("OOBE: ", 1 - forest_model.oob_score_)

ada_model = AdaBoostClassifier()
ada_model.fit(X_train,y_train)
y_pred_test_ada = ada_model.predict(X_test)
y_proba_ada = ada_model.predict_proba(X_test)
fpr_ada, tpr_ada, thresholds_ada = roc_curve(y_test, y_proba_ada[:, 1])
roc_curve_show("ADA",fpr_ada, tpr_ada)
accuracy_precision_recall("ADA",y_test,y_pred_test_ada)

gradient_model = GradientBoostingClassifier()
gradient_model.fit(X_train,y_train)
y_pred_test_gradient = gradient_model.predict(X_test)
y_proba_gradient = gradient_model.predict_proba(X_test)
fpr_gradient, tpr_gradient, thresholds_gradient = roc_curve(y_test, y_proba_gradient[:, 1])
roc_curve_show("GRADIENT",fpr_gradient, tpr_gradient)
accuracy_precision_recall("GRADIENT",y_test,y_pred_test_gradient)

