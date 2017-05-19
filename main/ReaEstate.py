import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

class Apartment:
    def __init__(self, data):
        self.data = data

    def getData(self):
        # Get data
        df = pd.read_csv(self.data, header=0)

        X_1 = df[['geo_lat', 'geo_long', 'number_of_rooms', 'number_of_toilets', 'surface_size']]
        X_2 = df[['location']]

        y = df[['price']].as_matrix()

        enc = preprocessing.OneHotEncoder()
        enc.fit(X_2)
        onehotlabels_1 = enc.transform(X_2).toarray()

        X = np.concatenate((X_1, onehotlabels_1), axis=1)
        return {'X': X, 'y':y}
    def splitData(self, test_size, X, y):
        # Split data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    def train(self, X_train, y_train):
        # Train
        mlp = MLPRegressor(hidden_layer_sizes=(100,100,100), max_iter=10000000)
        mlp.fit(X_train, y_train)
        print mlp.loss_
        return mlp

    def test(self, X_test, y_test, mlp):
        #Test
        predictions = mlp.predict(X_test)
        self.statistical(predictions, y_test)


    def statistical(self, predictions, y_test):
        # Check percent loss in test set
        loss = np.absolute(predictions - y_test)
        loss_percent = np.divide(loss, y_test)

        mean = np.sum(loss_percent)/loss_percent.shape
        print "mean " + str(mean)

        world = np.matrix.round(loss_percent, 1)
        test1 = loss_percent[np.where(world == 0.0)].shape[0]
        test2 = loss_percent[np.where(world == 0.1)].shape[0]
        test3 = loss_percent[np.where(world == 0.2)].shape[0]
        test4 = loss_percent[np.where(world >= 0.3)].shape[0]

        sizes = [test1, test2, test3, test4]
        labels = [0.0, 0.1, 0.2, 0.3]

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax1.axis('equal')
        plt.show()





