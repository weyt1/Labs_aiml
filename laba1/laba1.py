import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("test.csv")

print (df.head())
print (df.dtypes)
print (df.columns)

print(df)
nan_matrix = df.isnull()
print (nan_matrix)
print (nan_matrix.sum())
################## заполняем пропущенные значения
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())


no_numeric_cols =  df.select_dtypes(exclude='number').columns
for col in no_numeric_cols:
    if df[col].isnull().sum() > 0:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0])

nan_matrix = df.isnull()
print(nan_matrix.sum())
####################### нормализация
scaler = MinMaxScaler()

print(df[numeric_cols])
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(df[numeric_cols])

###################### преобразование категориальных данных
df = pd.get_dummies(df, columns=no_numeric_cols, drop_first=True)
print(df)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(train_df)
print (test_df)

df.to_csv("processed_titanic.csv", index=False)









