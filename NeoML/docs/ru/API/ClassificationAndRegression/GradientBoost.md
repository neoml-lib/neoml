# Градиентный бустинг CGradientBoost

<!-- TOC -->

- [Градиентный бустинг CGradientBoost](#градиентный-бустинг-cgradientboost)
	- [Параметры построения модели](#параметры-построения-модели)
		- [Функция потерь](#функция-потерь)
		- [Метод построения](#метод-построения)
	- [Модель](#модель)
		- [Для классификации](#для-классификации)
		- [Для регрессии](#для-регрессии)
	- [Модель QuickScorer](#модель-quickscorer)
		- [QuickScorer для классификации](#quickscorer-для-классификации)
		- [QuickScorer для регрессии](#quickscorer-для-регрессии)
	- [Пример](#пример)

<!-- /TOC -->

Метод градиентного бустинга строит ансамбль (комитет) решающих деревьев, предназначенный для решения задачи классификации или регрессии.
Для построения каждого дерева используется некоторое случайное подмножество элементов выборки и случайное подмножество признаков.

В **NeoML** алгоритм реализован классом `CGradientBoost`. Он предоставляет методы `Train` для обучения модели классификации и `TrainRegression` для обучения модели регрессии.

Алгоритм работает только с непрерывными признаками. В случае дискретных признаков необходимо сначала преобразовать их в непрерывные (например, используя бинаризацию).

## Параметры построения модели

Параметры реализованы структурой `CGradientBoost::CParams`.

- *LossFunction* — функция потерь;
- *IterationsCount* — максимальное количество итераций (количество деревьев в ансамбле);
- *LearningRate* — дополнительный множитель для каждого классификатора;
- *Subsample* — доля векторов, участвующая в построении одного дерева; может принимать значения из интервала [0..1];
- *Subfeature* — доля признаков, участвующая в построении одного дерева; может принимать значения из интервала [0..1];
- *Random* — генератор случайных чисел для выбора *Subsample* векторов и *Subfeature* признаков из всех;
- *MaxTreeDepth* — максимальная глубина каждого дерева;
- *MaxNodesCount* — максимальное количество вершин каждого деревa (при `-1` количество вершин не ограничено);
- *L1RegFactor* — параметр L1 регуляризации;
- *L2RegFactor* — параметр L2 регуляризации;
- *PruneCriterionValue* — значение разности критериев, при котором происходит склеивание вершин (при `0` склеивание не будет происходить никогда);
- *ThreadCount* — количество потоков, которое можно использовать во время обучения;
- *TreeBuilder* — тип построителя деревьев (*GBTB_Full* или *GBTB_FastHist*, см. [ниже](#метод-построения));
- *MaxBins* — максимальный размер гистограммы, используемый в режиме *GBTB_FastHist*;
- *MinSubsetWeight* — минимальный вес поддерева (`0` — без ограничений).

Параметры *L1RegFactor*, *L2RegFactor*, *PruneCriterionValue* применяются
к величинам, зависящим от суммы весов векторов в соответствующих вершинах дерева. Поэтому оптимальные значения этих параметров следует подбирать с учётом весов и количества векторов в вашей обучающей выборке.

### Функция потерь

При построении модели можно использовать следующие функции потерь:

- *LF_Exponential* — [только для классификации] экспоненциальная функция потерь: `L(x, y) = exp(-(2y - 1) * x)`;
- *LF_Binomial* — [только для классификации] биномиальная функция потерь: `L(x, y) = ln(1 + exp(-x)) - x * y`;
- *LF_SquaredHinge* — [только для классификации] сглаженный квадратичный Hinge: `L(x, y) = max(0, 1 - (2y - 1)* x) ^ 2`;
- *LF_L2* — квадратичная функция потерь: `L(x, y) = (y - x)^2 / 2`.

### Метод построения

Градиентный бустинг поддерживает четыре метода построения деревьев ансамбля:

- *GBTB_Full* — в качестве значений признаков для разделения перебираются все имеющиеся в задаче значения признаков.
- *GBTB_FastHist* — в качестве значений, используемых для разделения, будут взяты шаги гистограммы, построенной на значениях признаков. Параметр *MaxBins* задает размер гистограмм.
- *GBTB_MultiFull* — аналогично *GBTB_Full*, но в процессе строится не отдельное дерево для каждого целевого значения, а строится мультиклассовое дерево, листья которого содержат сразу вектор значений.
- *GBTB_MultiFastHist* — аналогично *GBTB_FastHist*, но с мультиклассовыми деревьями как в *GBTB_MultiFull*.


## Модель

В результате работы алгоритма строятся модели, описываемые интерфейсами `IGradientBoostModel` для классификации и `IGradientBoostRegressionModel` для регрессии.

### Для классификации

```c++
// Интерфейс модели, построенной градиентным бустингом
class NEOML_API IGradientBoostModel : public IModel {
public:
	virtual ~IGradientBoostModel() = 0;

	// Получить комитет
	virtual const CArray<CGradientBoostEnsemble>& GetEnsemble() const = 0;

	// Сериализация
	virtual void Serialize( CArchive& ) = 0;

	// Получение learning rate
	virtual double GetLearningRate() const = 0;

	// Получение функции потерь
	virtual CGradientBoost::TLossFunction GetLossFunction() const = 0;

	// Получить результаты классификации на всех подмножествах деревьев вида [1..k]
	virtual bool ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const = 0;
	virtual bool ClassifyEx( const CFloatVectorDesc& data, CArray<CClassificationResult>& results ) const = 0;

	// Посчитать статистику для признаков.
	// Возвращает число раз, которое данный признак был использован для разделения в деревьях.
	virtual void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const = 0;

	// Обрезает количество деревьев в модели в каждом комитете до данного
	virtual void CutNumberOfTrees( int numberOfTrees ) = 0;
};
```

### Для регрессии

```c++
// Интерфейс модели регрессии, построенной градиентным бустингом
class NEOML_API IGradientBoostRegressionModel : public IRegressionModel, public IMultivariateRegressionModel {
public:
	virtual ~IGradientBoostRegressionModel() = 0;
	
    // Получить комитет
	virtual const CArray<CGradientBoostEnsemble>& GetEnsemble() const = 0;

	// Сериализация
	virtual void Serialize( CArchive& ) = 0;

	// Получение learning rate
	virtual double GetLearningRate() const = 0;

	// Получение функции потерь
	virtual CGradientBoost::TLossFunction GetLossFunction() const = 0;

	// Посчитать статистику для признаков.
	// Возвращает число раз, которое данный признак был использован для разделения в деревьях.
	virtual void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const = 0;
};
```

## Модель QuickScorer

Также в библиотеке реализован алгоритм ускорения предсказания обученной модели, называемый [QuickScorer](http://ecmlpkdd2017.ijs.si/papers/paperID718.pdf).

Оптимизированная данным методом модель градиентного бустинга в некоторых случаях ускоряет предсказание до 10 раз.

Для оптимизации модели предназначен класс `CGradientBoostQuickScorer`:

```c++
// Механизм построения оптимизированной модели методом QuickScorer.
class NEOML_API CGradientBoostQuickScorer {
public:
	// Построить модель IGradientBoostQSModel на основе IGradientBoostModel.
	CPtr<IGradientBoostQSModel> Build( const IGradientBoostModel& gradientBoostModel );

	// Построить модель IGradientBoostQSRegressionModel на основе IGradientBoostRegressionModel.
	CPtr<IGradientBoostQSRegressionModel> BuildRegression( const IGradientBoostRegressionModel& gradientBoostModel );
};
```
Построенные классом модели реализуют интерфейсы `IGradientBoostQSModel`, `IGradientBoostQSRegressionModel`.

### QuickScorer для классификации

```c++
// Интерфейс оптимизированной модели.
class NEOML_API IGradientBoostQSModel : public IModel {
public:
	virtual ~IGradientBoostQSModel();
    
	// Сериализация.
	virtual void Serialize( CArchive& ) = 0;

	// Получение learning rate
	virtual double GetLearningRate() const = 0;

	// Получить результаты классификации на всех подмножествах деревьев вида [0..k].
	virtual bool ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const = 0;
	virtual bool ClassifyEx( const CFloatVectorDesc& data, CArray<CClassificationResult>& results ) const = 0;
};
```

### QuickScorer для регресcии

```c++
// Интерфейс оптимизированной модели регрессии.
class IGradientBoostQSRegressionModel : public IRegressionModel {
public:
	// Сериализация.
	virtual void Serialize( CArchive& ) = 0;

	// Получение learning rate
	virtual double GetLearningRate() const = 0;
};
```

## Пример

Ниже представлен простой пример обучения модели градиентного бустинга. Входные данные подаются в виде объекта, реализующего интерфейс [`IProblem`](Problems.md).

```c++
CPtr<IModel> buildModel( IProblem* data )
{
	CGradientBoost::CParams params;
	params.LossFunction = CGradientBoost::LF_Exponential;
	params.IterationsCount = 100;
	params.LearningRate = 0.1;
	params.MaxTreeDepth = 10;
	params.ThreadCount = 4;
	params.Subsample = 0.5;
	params.Subfeature = 1;
	params.MaxBins = 64;
	params.TreeBuilder = GBTB_FastHist;
	params.MinSubsetWeight = 10;

	CGradientBoost boosting( params );
	return boosting.Train( *problem );
}
```
