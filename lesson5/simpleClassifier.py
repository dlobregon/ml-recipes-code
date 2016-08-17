import random
class ScrappyKNN():
    # first, define a fit method for training
    def fit(self, X_train, y_train):
        #creating a simple random classifier, in other words, only guess the label
        self.X_train = X_train
        self.y_train = y_train
    # predict method for testing data, the output is a label prediction
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

from sklearn import datasets
iris = datasets.load_iris()

#defining the two parts like a fuction
X = iris.data
Y = iris.target

#impor training data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .5)

#creating our ScrappyKNN
my_classifier = ScrappyKNN()
#training the classifier using the training data
my_classifier.fit(X_train, y_train)

#now we call the predict method to work with our test data
predictions = my_classifier.predict(X_test)
print "the predictions of our KNeighborsClassifier are:"

print predictions

#importing the metrics methods to know the accuracy of our decision KNeighborsClassifier
from sklearn.metrics import accuracy_score
print ""
print "the accuracy of our KNeighborsClassifier is:"
print accuracy_score(y_test,predictions)
