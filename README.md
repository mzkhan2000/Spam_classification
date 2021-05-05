# Spam_classification
Spam classification
Our Mission
You recently used Naive Bayes to classify spam in this dataset. In this notebook, we will expand on the previous analysis by using a few of the new techniques you've learned throughout this lesson.

Let's quickly re-create what we did in the previous Naive Bayes Spam Classifier notebook. We're providing the essential code from that previous workspace here, so please run this cell below.

Turns Out...
We can see from the scores above that our Naive Bayes model actually does a pretty good job of classifying spam and "ham." However, let's take a look at a few additional models to see if we can't improve anyway.

Specifically in this notebook, we will take a look at the following techniques:

BaggingClassifier
RandomForestClassifier
AdaBoostClassifier
Another really useful guide for ensemble methods can be found in the documentation here.

These ensemble methods use a combination of techniques you have seen throughout this lesson:

Bootstrap the data passed through a learner (bagging).
Subset the features used for a learner (combined with bagging signifies the two random components of random forests).
Ensemble learners together in a way that allows those that perform best in certain areas to create the largest impact (boosting).
In this notebook, let's get some practice with these methods, which will also help you get comfortable with the process used for performing supervised machine learning in Python in general.

Since you cleaned and vectorized the text in the previous notebook, this notebook can be focused on the fun part - the machine learning part.

This Process Looks Familiar...
In general, there is a five step process that can be used each time you want to use a supervised learning method (which you actually used above):

Import the model.
Instantiate the model with the hyperparameters of interest.
Fit the model to the training data.
Predict on the test data.
Score the model by comparing the predictions to the actual values.
Follow the steps through this notebook to perform these steps using each of the ensemble methods: BaggingClassifier, RandomForestClassifier, and AdaBoostClassifier.

Step 1: First use the documentation to import all three of the models.

# Import the Bagging, RandomForest, and AdaBoost Classifier
​
Step 2: Now that you have imported each of the classifiers, instantiate each with the hyperparameters specified in each comment. In the upcoming lessons, you will see how we can automate the process to finding the best hyperparameters. For now, let's get comfortable with the process and our new algorithms.

# Instantiate a BaggingClassifier with:
# 200 weak learners (n_estimators) and everything else as default values
​
​
​
# Instantiate a RandomForestClassifier with:
# 200 weak learners (n_estimators) and everything else as default values
​
​
# Instantiate an a AdaBoostClassifier with:
# With 300 weak learners (n_estimators) and a learning_rate of 0.2
​
​
Step 3: Now that you have instantiated each of your models, fit them using the training_data and y_train. This may take a bit of time, you are fitting 700 weak learners after all!

# Fit your BaggingClassifier to the training data
​
​
# Fit your RandomForestClassifier to the training data
​
​
# Fit your AdaBoostClassifier to the training data
​
​
Step 4: Now that you have fit each of your models, you will use each to predict on the testing_data.

# Predict using BaggingClassifier on the test data
​
​
# Predict using RandomForestClassifier on the test data
​
​
# Predict using AdaBoostClassifier on the test data
​
​
Step 5: Now that you have made your predictions, compare your predictions to the actual values using the function below for each of your models - this will give you the score for how well each of your models is performing. It might also be useful to show the Naive Bayes model again here, so we can compare them all side by side.

def print_metrics(y_true, preds, model_name=None):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (NumPy array or pandas series)
    preds - the predictions for those values from some model (NumPy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements 
    
    OUTPUT:
    None - prints the accuracy, precision, recall, and F1 score
    '''
    if model_name == None:
        print('Accuracy score: ', format(accuracy_score(y_true, preds)))
        print('Precision score: ', format(precision_score(y_true, preds)))
        print('Recall score: ', format(recall_score(y_true, preds)))
        print('F1 score: ', format(f1_score(y_true, preds)))
        print('\n\n')
    
    else:
        print('Accuracy score for ' + model_name + ' :' , format(accuracy_score(y_true, preds)))
        print('Precision score ' + model_name + ' :', format(precision_score(y_true, preds)))
        print('Recall score ' + model_name + ' :', format(recall_score(y_true, preds)))
        print('F1 score ' + model_name + ' :', format(f1_score(y_true, preds)))
        print('\n\n')
# Print Bagging scores
​
​
# Print Random Forest scores
​
​
# Print AdaBoost scores
​
​
# Naive Bayes Classifier scores
​
​
Recap
Now you have seen the whole process for a few ensemble models!

Import the model.
Instantiate the model with the hyperparameters of interest.
Fit the model to the training data.
Predict on the test data.
Score the model by comparing the predictions to the actual values.
And that's it. This is a very common process for performing machine learning.

But, Wait...
You might be asking -

What do these metrics mean?

How do I optimize to get the best model?

There are so many hyperparameters to each of these models, how do I figure out what the best values are for each?

This is exactly what the last two lessons of this course on supervised learning are all about.

Notice, you can obtain a solution to this notebook by clicking the orange icon in the top left!
