# Multi-Class Classification Sample

<!-- TOC -->

- [Multi-Class Classification Sample](#multi-class-classification-sample)
	- [Preparing the input data](#preparing-the-input-data)
	- [Training the classifier](#training-the-classifier)
	- [Analyzing the results](#analyzing-the-results)

<!-- /TOC -->

This tutorial walks through training **NeoML** classification model to classify the well-known [News20](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) data set.

We are going to use the combination of [linear classifier](../API/ClassificationAndRegression/Linear.md) and the ["one versus all"](../API/ClassificationAndRegression/OneVersusAll.md) method.

## Preparing the input data

We assume that the data set is split into two parts: *train* and *test*, and each is serialized in a file on disk as a `CMemoryProblem` (which is a simple implementation of the `IProblem` interface provided in the library).

The library serialization methods can be used to load the data into memory for processing.

```c++
CPtr<CMemoryProblem> trainData = new CMemoryProblem();
CPtr<CMemoryProblem> testData = new CMemoryProblem();

CArchiveFile trainFile( "news20.train", CArchive::load );
CArchive trainArchive( &trainFile, CArchive::load );
trainArchive >> trainData;

CArchiveFile testFile( "news20.test", CArchive::load );
CArchive testArchive( &testFile, CArchive::load );
testArchive >> testData;
```

## Training the classifier

The "one versus all" classifier uses the specified classifier to train a model per each class that would determine the probability for an object to belong to this class. An input object is then classified by the models voting.

1. Create a linear classifier using the `CLinearClassifier` class (`COneVersusAll` will take place implicitly). Select the logistic regression loss function (`EF_LogReg` constant).
2. Call the `Train` method, passing the `trainData` training set prepared above. The method will train the model and return it as an object implementing the `IModel` interface.

```c++
CLinearClassifier linear( EF_LogReg );
CPtr<IModel> model = linear.Train( *trainData );
```

## Analyzing the results

We can check the results the trained model shows on the test sample using the `Classify` method of the `IModel` interface. Call this method for each vector of the `testData` data set prepared before.

```c++
int correct = 0;
for( int i = 0; i < testData->GetVectorCount(); i++ ) {
	CClassificationResult result;
	model->Classify( testData->GetVector( i ), result );

	if( result.PreferredClass == testData->GetClass( i ) ) {
		correct++;
	}
}

double totalResult = static_cast<double>(correct) / testData->GetVectorCount();
printf("%.3f\n", totalResult);
```

On this testing run, 83.3% of the vectors were classified correctly.

```
0.833
```