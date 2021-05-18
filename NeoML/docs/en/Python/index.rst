.. NeoML documentation master file, created by
   sphinx-quickstart on Wed Jan 20 20:52:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#################################
Welcome to NeoML documentation!
#################################

.. image:: ../../images/NeoML_logo_help.png

`neoml` module provides a Python interface for the `C++ NeoML library <https://github.com/neoml-lib/neoml>`_.

NeoML is an end-to-end machine learning framework that allows you to build, train, and deploy ML models. This framework is used by ABBYY engineers for computer vision and natural language processing tasks, including image preprocessing, classification, document layout analysis, OCR, and data extraction from structured and unstructured documents.

Basic principles
###################

Platform independence
***********************

The user interface is completely separated from the low-level calculations implemented by a math engine. The only thing you have to do is to specify at the start the type of the math engine that will be used for calculations. You can also choose to select the math engine automatically, based on the device configuration detected.

The rest of your machine learning code will be the same regardless of the math engine you choose.

Math engines independence
********************************

Each network works with one math engine instance, and all its layers should have been created with the same math engine. If you have chosen a GPU math engine, it will perform all calculations. This means you may not choose to use a CPU for "light" calculations like adding vectors and a GPU for "heavy" calculations like multiplying matrices. We have introduced this restriction to avoid unnecessary synchronizations and data exchange between devices.

Multi-threading support
***************************

The math engine interface is thread-safe; the same instance may be used in different networks and different threads. Note that this may entail some synchronization overhead.

However, the :ref:`neural network implementation <py-submodule-dnn>` is not thread-safe; the network may run only in one thread.

.. ONNX support
   ********************

   **NeoML** library also works with the models created by other frameworks, as long as they support the `ONNX <https://onnx.ai/>` format. See the description of import API. However, you cannot export a NeoML-trained model into ONNX format.


Submodules
##########

.. toctree::
   :maxdepth: 3
   :caption: Submodules
   :hidden:

   submodules/dnn
   submodules/clustering
   submodules/classificationregression
   submodules/mathengine
   submodules/misc



The Python API contains several submodules:

- :ref:`Neural networks <py-submodule-dnn>`
- :ref:`Clustering algorithms <py-submodule-clustering>`
- :ref:`Classification and regression algorithms <py-submodule-classificationregression>`
- :ref:`Math engine used for calculations <py-submodule-mathengine>`
- :ref:`Other algorithms <py-submodule-misc>`

Tutorials
#########

.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   :hidden:

   tutorials/Cifar10
   tutorials/IRNN
   tutorials/Linear
   tutorials/KMeans
   tutorials/Regressor
   tutorials/Boosting

Here are several guides that walk you through using NeoML for simple practical tasks, illustrating the specifics of building NeoML networks, working with the blob data format, and evaluating the performance.

- :ref:`Neural network for CIFAR-10 dataset <tutorials/Cifar10.ipynb>`
- :ref:`Identity recurrent neural network (IRNN) <tutorials/IRNN.ipynb>`
- :ref:`Linear classifier <tutorials/Linear.ipynb>`
- :ref:``k`-means clustering <tutorials/KMeans.ipynb>`
- :ref:`Linear regressor <tutorials/Regressor.ipynb>`
- :ref:`Gradient boosting classifier <tutorials/Boosting.ipynb>`

Installation
############

.. toctree::
   :maxdepth: 3
   :caption: Installation
   :hidden:

Install a stable version of NeoML library from PyPI::

    pip3 install neoml

Install the library you downloaded locally from `our github repo <https://github.com/neoml-lib/neoml>`_::

    cd <path to neoml>/NeoML/Python
    python3 setup.py install

Supported Python versions: 3.6 to 3.9


Index
##################

* :ref:`genindex`
* :ref:`search`
