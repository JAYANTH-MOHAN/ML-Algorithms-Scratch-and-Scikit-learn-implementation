import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train=pd.read_csv('train.csv')
test = pd.read_csv("test.csv")


train = train.drop(["Unnamed: 0", "Id"], axis = 1)
test = test.drop(["Unnamed: 0", "Id"], axis = 1)
train


train_data = train.values
Y = train_data[:, -1].reshape(train_data.shape[0], 1)
X = train_data[:, :-1]


test_data = test.values
Y_test = test_data[:, -1].reshape(test_data.shape[0], 1)
X_test = test_data[:, :-1]


print("Shape of X_train :", X.shape)
print("Shape of Y_train :", Y.shape)
print("Shape of X_test :", X_test.shape)
print("Shape of Y_test :", Y_test.shape)


X = np.vstack((np.ones((X.shape[0], )), X.T)).T
X_test = np.vstack((np.ones((X_test.shape[0], )), X_test.T)).T


def model(X, Y, learning_rate, iteration):
    m = Y.size
    theta = np.zeros((X.shape[1], 1))
    cost_list = []
    for i in range(iteration):
        y_pred = np.dot(X, theta)
        cost = (1/(2*m))*np.sum(np.square(y_pred - Y))
        d_theta = (1/m)*np.dot(X.T, y_pred - Y)
        theta = theta - learning_rate*d_theta
        cost_list.append(cost)
    # to print the cost for 10 times
        if(iget_ipython().run_line_magic("(iteration/10)", " == 0):")
            print("Cost is :", cost)
    return theta, cost_list


iteration = 10000
learning_rate = 0.000000005
theta, cost_list = model(X, Y, learning_rate = learning_rate, iteration =iteration)


rng = np.arange(0, iteration)
plt.plot(rng, cost_list)
plt.show()


y_pred = np.dot(X_test, theta)
error = (1/X_test.shape[0])*np.sum(np.abs(y_pred - Y_test))


print("Test error is :", error*100, "get_ipython().run_line_magic("")", "")
print("Test Accuracy is :", (1- error)*100, "get_ipython().run_line_magic("")", "")


plt.plot(Y_test[0:3,0:3],y_pred[0:3:,0:3])



