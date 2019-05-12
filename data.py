#matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set_1 = pd.read_csv('data_set1.csv')
data_set_2 = pd.read_csv('data_set2.csv')
#combining the data sets
data_set = pd.concat([data_set_1 , data_set_2],axis=0, join='outer', ignore_index=False)
print(data_set.head())


#droping student coloum
data_set.drop(["No_of_Student"], axis = 1, inplace = True)
#droping date_time  & Serial number coloum
data_set.drop(["SL_NO"], axis = 1, inplace = True)

#data_set.info(memory_usage='deep')
plt.plot(data_set['DATE_TIME'], data_set['PM2.5_B'], 'ro')
#plt.show()

PMtag = []
for i in data_set['PM2.5_B']:
	if i<=100:
		PMtag.append(0)
	if i<=200:
		PMtag.append(1) 	
	else:
		PMtag.append(2)	
			
#print(PMtag)

PMtagdf = pd.DataFrame({'PMtag':PMtag})
#print(PMtagdf.head())

data_set['PM2.5_B'] = PMtagdf['PMtag']
print(data_set.head())
#data_set.info(memory_usage='deep')

#handling NULL data
data_set.fillna(data_set.mean())
#handling error data
for coloum in data_set:
	for i in data_set[coloum]:
	 	if (type(i) !=float and type(i) != int ):
	 		data_set.drop([coloum], axis = 1, inplace = True)
	 		break
			
# print(data_set.shape)	
#data_set.info(memory_usage='deep')
#print(data_set.head())

#################################################
#classification for PM2.5_D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


X = data_set.loc[:, data_set.columns != 'PM2.5_B'].values

y = data_set.loc[: , ['PM2.5_B']].values

# # #standarising the features
# # X = StandardScaler().fit_transform(X)

# # pca = PCA(n_components = 2)
# # X = pca.fit_transform(X)
# # #print(x)
# # #print(y)

#plt.plot(x,y)
#plt.show()

# # #print(X.info)
# # #print(y.info)
# # principleDf_feature = pd.DataFrame(X)
# # principleDf_lebel = pd.DataFrame(y)

# # finalDf = pd.concat([principleDf_feature,principleDf_lebel], axis = 1)

# # finalDf.info(memory_usage='deep')

# # print(finalDf.head())
from sklearn import preprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,y_train.ravel())

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))


#
#using decesion tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))




##
#using k- nearest neighbour
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train.ravel())
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))