.. _py-submodule-classificationregression:

################
neoml.ClassificationRegression
################

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

Classifier class description
****************************

.. autoclass:: neoml.GradientBoostClassifier
   :members:

Classification model class description
**************************************

.. autoclass:: neoml.GradientBoostClassificationModel
   :members:

Regressor class description
***************************

.. autoclass::neoml.GradientBoostRegressor
   :members:

Regression model class description
**********************************

.. autoclass::neoml.GradientBoostRegressionModel
   :members:

Example
*******

.. code-block:: python

   import numpy as np
   import neoml

.. _py-classification-linear:
.. _py-classification-svm:
.. _py-classification-decisiontree:
.. _py-classification-onevsall:
