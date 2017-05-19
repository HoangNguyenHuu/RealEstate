import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

def chart(loss_percent):
    less_than_10 = 0
    less_than_20 = 0
    less_than_30 = 0
    rest = 0
    for i in range(0, loss_percent.shape[0]):
        if loss_percent[i] < 0.1:
            less_than_10 = less_than_10 + 1
        elif loss_percent[i] <= 0.2:
            less_than_20 = less_than_20 + 1
        elif loss_percent[i] <= 0.3:
            less_than_30 = less_than_30 + 1
        else:
            rest = rest + 1

    labels = ['<10%', '10-20%', '20-30%', '>30%']
    sizes = [less_than_10, less_than_20, less_than_30, rest]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax1.axis('equal')
    plt.show()

input_file = "../data/chungcu_ban.csv"
df = pd.read_csv(input_file, header=0)
X_1 = df[['geo_lat', 'geo_long', 'number_of_rooms', 'number_of_toilets', 'surface_size']]
X_2 = df[['location']]

y = df['price'].as_matrix()

enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
onehotlabels_1 = enc.transform(X_2).toarray()

X = np.concatenate((X_1, onehotlabels_1), axis=1)
print X.shape


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
print(X_train.shape)
print(y_train.shape)


mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=10000000)
mlp.fit(X_train, y_train)
print(mlp.loss_)

predictions = mlp.predict(X_test)
print(predictions[1:10])
print(y[1:10])

loss = np.absolute(predictions - y_test)
loss_percent = np.divide(loss, y_test)
# hello = loss_percent[np.where(loss_percent > 3)]
# hello_mean = np.sum(hello)/hello.shape
# loss_mean = np.sum(loss_percent)/loss_percent.shape
# print(loss_mean)


chart(loss_percent)
