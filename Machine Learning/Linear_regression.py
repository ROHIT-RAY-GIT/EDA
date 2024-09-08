import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\Data Science & AI\Jupyter Notebooks\Machine Learning\Data\Salary_Data.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set, x-train and y-train)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set, x-test and y-test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
m_slope


intercept = regressor.intercept_
intercept


y_15 = m_slope * 15 + intercept
y_15


