.. _py-submodule-classificationregression:

##############################
neoml.ClassificationRegression
##############################

In `neoml` module, you can find various methods for solving classification and regression problems.

- :ref:`py-classification-gradientboosting`
- :ref:`py-classification-linear`
- :ref:`py-classification-svm`
- :ref:`py-classification-decisiontree`
- :ref:`py-classification-onevsall`

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

.. autoclass::neoml.GradientBoost.GradientBoostRegressor
   :members:

Regression model
**********************************

.. autoclass::neoml.GradientBoost.GradientBoostRegressionModel
   :members:

.. _py-classification-linear:

Linear
#################

A linear classifier finds a hyperplane that divides the feature space in half.

.. autoclass::neoml.LinearClassifier
   :members:

.. autoclass::neoml.LinearClassificationModel
   :members:

.. autoclass::neoml.LinearRegressor
   :members:

.. autoclass::neoml.LinearRegressionModel

.. _py-classification-svm:

Support-vector machine
######################

.. _py-classification-decisiontree:

Decision tree
#############

.. _py-classification-onevsall:

One versus all
##############