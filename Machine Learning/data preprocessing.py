# nd array
import numpy as np 	#Array		
# visualization
import matplotlib.pyplot as plt		
# dataframe
import pandas as pd	

# read the dataset 
dataset = pd.read_csv(r"C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\1. KODI WORK\1. NARESH\2. EVENING BATCH\N_Batch -- 7.30PM\3. Sep\3rd - ML\5. Data preprocessing\Data.csv")
# indepdendent variable
X = dataset.iloc[:, :-1].values	
#dependent variable
y = dataset.iloc[:,3].values  

#
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer() 

imputer = imputer.fit(X[:,1:3]) 

X[:, 1:3] = imputer.transform(X[:,1:3])


from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0]) 

X[:,0] = labelencoder_X.fit_transform(X[:,0]) 

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)