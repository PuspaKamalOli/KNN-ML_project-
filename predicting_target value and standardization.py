import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# preprocessing
df = pd.read_csv('Classified Data', index_col=0)
df.head()

# standardizing data to make mean=0 and sd=1
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

# training data and model building
x = df_feat
y = df['TARGET CLASS']
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(x_test)

# testing report
classification_report(y_test, pred)
confusion_matrix(y_test, pred)

# elbow method to check which is best value for k
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', marker='o',
         markerfacecolor='red', ls='dashed', markersize=10)
plt.xlabel('K')
plt.ylabel('error rate')
plt.title('k vs error rate')
