from ReaEstate import Apartment

apartment = Apartment('../data/chungcu_ban.csv')
data = apartment.getData()
X = data['X']
y = data['y']

data_set = apartment.splitData(0.1, X,y)
X_train = data_set['X_train']
y_train = data_set['y_train']
X_test = data_set['X_test']
y_test = data_set['y_test']

mlp = apartment.train(X_train, y_train)
apartment.test(X_test,y_test, mlp)