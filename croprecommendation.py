import numpy as np
import matplotlib as plt 
import pandas as pd 


data = pd.read_csv("Crop_recommendation.csv")
print(data)
print(data)
X=data.iloc[: , 2:7]
print(X)
Y=data.iloc[: , -1]
print(Y)
print(X)
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train  , Y_test = train_test_split(X ,Y , test_size=0.2 )
from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train , Y_train )
classifier_en =DecisionTreeClassifier(criterion='entropy')
classifier_en.fit(X_train , Y_train )
classifier.score(X_test , Y_test)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

sc.fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)
classifier_sc= DecisionTreeClassifier(criterion='gini')
classifier_sc.fit(X_train_sc , Y_train)
classifier_sc.score(X_test_sc , Y_test)
crop1=[3,122,0,7,300]
crop1=np.array([crop1])
crop1
classifier.predict(crop1)
pred=classifier.predict(crop1)
print(pred[0])
if pred[0] == 0 :
  print('You Should Harvest Rice ')
elif pred[0] == 1:
  print('You Should Harvest wheat ')
import matplotlib.pyplot as plt 
import seaborn 
import pandas as pd 
data = pd.read_csv("Crop_recommendation.csv") 
data.head()

seaborn.scatterplot( data=data, x="N", y="P", hue="label") 
plt.xlabel("percentage of nitrogen in the soil") 
plt.ylabel("percentage of phosphorus in the soil") 
plt.title("crop recommendation visualization")
seaborn.scatterplot( data=data, x="temperature", y="rainfall", hue="label", palette="deep")
plt.xlabel("Temprature")
plt.ylabel("Rainfall")
plt.title("crop  recommendation visualization")