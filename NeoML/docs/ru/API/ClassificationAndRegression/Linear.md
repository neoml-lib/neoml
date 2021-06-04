# Линейный классификатор CLinear

<!-- TOC -->

- [Линейный классификатор CLinear](#линейный-классификатор-clinear)
	- [Параметры построения модели](#параметры-построения-модели)
		- [Функция потерь](#функция-потерь)
	- [Модель](#модель)
		- [Для классификации](#для-классификации)
		- [Для регрессии](#для-регрессии)
	- [Пример](#пример)

<!-- /TOC -->

Линейный классификатор — алгоритм классификации, основанный на построении линейной разделяющей поверхности, которая делит пространство признаков на два полупространства.

В **NeoML** алгоритм реализован классом `CLinear`. Он предоставляет методы `Train` для обучения модели классификации и `TrainRegression` для обучения модели линейной регрессии.

## Параметры построения модели

Параметры реализованы структурой `CLinear::CParams`.

- *Function* — функция потерь;
- *MaxIterations* — максимальное количество итераций;
- *ErrorWeight* — вес ошибок относительно регуляризатора;
- *SigmoidCoefficients* — предопределенная сигмоида;
- *Tolerance* — критерий останова;
- *NormalizeError* — указывает, необходима ли нормализация ошибки;
- *L1Coeff* — коэффициент L1 регуляризации; чтобы использовать L2-регуляризацию, присвойте ему значение `0`;
- *ThreadCount* — количество потоков, которое можно использовать во время обучения;
- *MulticlassMode* - подход, используемый при многоклассовой классификации: OneVsAll (по умолчанию) или OneVsOne.

### Функция потерь

Доступные функции потерь для оценки результата:

- *EF_SquaredHinge* — квадратичный Hinge;
- *EF_LogReg* — функция логистической регрессии;
- *EF_SmoothedHinge* — половина гиперболы;
- *EF_L2_Regression* — функция L2 для задачи регрессии.

## Модель

### Для классификации

```c++
// Интерфейс модели классификатора.
class NEOML_API ILinearBinaryModel : public IModel {
public:
	virtual ~ILinearBinaryModel() = 0;

	// Получить разделяющую плоскость.
	virtual CFloatVector GetPlane() const = 0;

	// Сериализация.
	virtual void Serialize( CArchive& ) = 0;
};
```

### Для регрессии

```c++
// Интерфейс модели регрессии.
class NEOML_API ILinearRegressionModel : public IRegressionModel {
public:
	virtual ~ILinearRegressionModel() = 0;

	// Получить разделяющую плоскость.
	virtual CFloatVector GetPlane() const = 0;

	// Сериализация.
	virtual void Serialize( CArchive& ) = 0;
};
```

## Пример

Ниже представлен простой пример обучения линейного классификатора. Входные данные подаются в виде объекта, реализующего интерфейс [`IProblem`](Problems.md).

```c++
CPtr<Model> buildModel( IProblem* data )
{
	CLinear::CParams params;
	params.Function = EF_SquaredHinge;
	params.L1Coeff = 0.05;
	params.ThreadCount = 4;

	CLinear builder( params );
	return builder.Train( *data );
}
```