# Пример многоклассовой классификации

<!-- TOC -->

- [Пример многоклассовой классификации](#пример-многоклассовой-классификации)
	- [Подготовка входных данных](#подготовка-входных-данных)
	- [Обучение классификатора](#обучение-классификатора)
	- [Анализ результата](#анализ-результата)

<!-- /TOC -->

В этом примере мы используем библиотеку **NeoML** для обучения модели, решающей задачу многоклассовой классификации на классическом датасете [News20](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups).

Для решения задачи используем комбинацию из двух методов: [линейного бинарного классификатора](../API/ClassificationAndRegression/Linear.md)
и метода ["один против всех"](../API/ClassificationAndRegression/OneVersusAll.md).

## Подготовка входных данных

Предположим, что датасет состоит из двух частей *train* и *test*, каждая из которых сериализована в файле в виде `CMemoryProblem`, простейшей реализации интерфейса `IProblem`.

Используем средства сериализации библиотеки для загрузки данных.
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

## Обучение классификатора

Классификатор "один против всех" использует заданный классификатор для того, чтобы на каждый класс обучить модель, определяющую вероятность принадлежности объекта к этому классу. Отнесение объекта к одному из классов затем определяется путём голосования моделей.

1. Создадим линейный классификатор при помощи класса `CLinearClassifier`. В качестве функции потерь используем логистическую регрессию (константа `EF_LogReg`).
2. Вызовем метод `Train`, который получает на вход обучающую выборку `trainData`, подготовленную на предыдущем шаге. Этот метод обучит модель и вернёт её в виде объекта, реализующего интерфейс `IModel`.

```c++
CLinearClassifier linear( EF_LogReg );
CPtr<IModel> model = oneVersusAll.Train( *trainData );
```

## Анализ результата

Проверим результаты работы обученной модели на тестовой выборке. Для этого `IModel` предоставляет метод `Classify`; вызовем его для каждого вектора из ранее подготовленной выборки `testData`.

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

В нашем тестовом запуске 83.3% векторов были классифицированы верно.

```
0.833
```