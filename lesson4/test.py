from sklearn import datasets
iris = datasets.load_iris()

#defining the two parts like a fuction
X = iris.data
Y = iris.target

#impor training data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .5)

#import our first classifier
from sklearn import tree
my_classifier= tree.DecisionTreeClassifier()

#training the classifier using the training data
my_classifier.fit(X_train, y_train)

#now we call the predict method, to predict and work with our test data
predictions = my_classifier.predict(X_test)
print "predictions are:"
print predictions


#importing the metrics methods to know the accuracy of our decision tree
from sklearn.metrics import accuracy_score
print ""
print "the accuracy of our tree is:"
print accuracy_score(y_test,predictions)
