import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#данные
df = pd.read_csv('house_price_regression_dataset.csv')
target = 'House_Price'
X = df.drop(columns=[target])
y = df[target]

#80% учим, 20% тесты
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Функция RMSE - корень из средней квадратичной ошибки
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate(name, y_true, y_pred):
    print(f"{name}: MSE={mean_squared_error(y_true, y_pred):.3f}, RMSE={rmse(y_true, y_pred):.3f}, MAE={mean_absolute_error(y_true, y_pred):.3f}, R2={r2_score(y_true, y_pred):.3f}")

#Экс 1: линейная регрессия
model1 = LinearRegression().fit(X_train, y_train)
evaluate("Exp1", y_test, model1.predict(X_test))

#Экс 2: экс 1 + стандартизация
scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train)
X_test2 = scaler2.transform(X_test)
model2 = LinearRegression().fit(X_train2, y_train)
evaluate("Exp2", y_test, model2.predict(X_test2))

#Экс 3: убираем гараж, смотрим насколько он лишний
X3 = X.drop(columns=['Garage_Size'])
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y, test_size=0.2, random_state=42)
model3 = LinearRegression().fit(X_train3, y_train3)
evaluate("Exp3", y_test3, model3.predict(X_test3))

#Экс 4: новые признаки - возраст дома и площадь/комнату (это на основе старых)
X4 = X.copy()
X4['Age'] = 2026 - X4['Year_Built']
X4['Area_per_Room'] = X4['Square_Footage'] / (X4['Num_Bedrooms'] + X4['Num_Bathrooms'] + 1)
X4 = X4.drop(columns=['Year_Built'])
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y, test_size=0.2, random_state=42)
scaler4 = StandardScaler()
X_train4 = scaler4.fit_transform(X_train4)
X_test4 = scaler4.transform(X_test4)
model4 = LinearRegression().fit(X_train4, y_train4)
evaluate("Exp4", y_test4, model4.predict(X_test4))

#Экс 5: ridge + стандартизация (без нее штрафам плохо😭)
scaler5 = StandardScaler()
X_train5 = scaler5.fit_transform(X_train)
X_test5 = scaler5.transform(X_test)
model5 = Ridge(alpha=0.5).fit(X_train5, y_train)
evaluate("Exp5", y_test, model5.predict(X_test5))