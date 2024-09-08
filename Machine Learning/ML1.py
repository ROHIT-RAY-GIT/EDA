import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv((r'D:\Data Science & AI\Jupyter Notebooks\Data\Data - Data.csv'))

x = dataset.iloc[:,: -1].values
y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

from sklearn.preprocessing import LabelEncoder

labelEncoder_x = LabelEncoder()
labelEncoder_x.fit_transform(x[:, 0])

x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, test_size = 0.2, random_state = 0)
