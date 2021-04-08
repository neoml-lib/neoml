.. _py-submodule-classificationregression:

##############################
neoml.ClassificationRegression
##############################

In `neoml` module, you can find various methods for solving classification and regression problems.

- :ref:`py-classification-gradientboosting`
- :ref:`py-classification-linear`
- :ref:`py-classification-svm`
- :ref:`py-classification-decisiontree`

.. _py-classification-gradientboosting:

Gradient tree boosting
######################

Gradient boosting method creates an ensemble of decision trees using random subsets of features and input data.
The algorithm only accepts continuous features. If your data is characterized by discrete features you will need to transform them into continuous ones (for example, using binarization).

Classifier
****************************

.. autoclass:: neoml.GradientBoost.GradientBoostClassifier
   :members:

Classification model
**************************************

.. autoclass:: neoml.GradientBoost.GradientBoostClassificationModel
   :members:

Regressor
***************************

.. autoclass:: neoml.GradientBoost.GradientBoostRegressor
   :members:

Regression model
**********************************

.. autoclass:: neoml.GradientBoost.GradientBoostRegressionModel
   :members:

.. _py-classification-linear:

Linear
#################

A linear classifier finds a hyperplane that divides the feature space in half.

.. autoclass:: neoml.Linear.LinearClassifier
   :members:

.. autoclass:: neoml.Linear.LinearClassificationModel
   :members:

.. autoclass:: neoml.Linear.LinearRegressor
   :members:

.. autoclass:: neoml.Linear.LinearRegressionModel

.. _py-classification-svm:

Support-vector machine
######################

Support-vector machine translates the input data into vectors in a high-dimensional space and searches for a maximum-margin dividing hyperplane.

.. autoclass:: neoml.SVM.SvmClassifier
   :members:

.. autoclass:: neoml.SVM.SvmClassificationModel
   :members:

.. _py-classification-decisiontree:

Decision tree
#############

Decision tree is a classification method that involves comparing the object features with a set of threshold values; the result tells us to move to one of the children nodes. Once we reach a leaf node we assign the object to the class this node represents.

.. autoclass:: neoml.DecisionTree.DecisionTreeClassifier
   :members:

.. autoclass:: neoml.DecisionTree.DecisionTreeClassificationModel
   :members: