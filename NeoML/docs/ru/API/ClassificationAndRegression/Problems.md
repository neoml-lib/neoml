# Интерфейсы входных данных для обучения

<!-- TOC -->

- [Интерфейсы входных данных для обучения](#интерфейсы-входных-данных-для-обучения)
	- [Для классификации](#для-классификации)
		- [Пример реализации](#пример-реализации)
	- [Для регрессии](#для-регрессии)

<!-- /TOC -->

Чтобы обучить модель для классификации или регрессии, вам необходимо представить свои входные данные в виде объекта, реализующего один из описанных ниже интерфейсов.

## Для классификации

Входные данные для обучения модели классификации должны быть представлены в виде интерфейса `IProblem`. Он должен содержать всю информацию для обучения модели. Основные данные — это набор векторов, каждый из которых содержит значения нескольких признаков для одного объекта. Кроме того, `IProblem` должен предоставлять общую информацию о классах и признаках, которые используются для классификации:

- *GetClassCount* — число классов, на которые будем классифицировать;
- *GetFeatureCount* — число признаков, которыми характеризуются объекты (т.е. длина одного вектора входных данных);
- *IsDiscreteFeature* — для каждого из признаков (по порядковому номеру) сообщает, принимает ли он лишь дискретные значения;
- *GetVectorCount* — число векторов в наборе, т.е. различных объектов, которые мы классифицируем;
- *GetClass* — класс, к которому принадлежит вектор с заданным порядковым номером; классы обозначаются номерами от 0 до (*GetClassCount* - 1);
- *GetVector* — вектор с заданным порядковым номером;
- *GetMatrix* — все вектора набора в виде матрицы (размером *GetFeatureCount* * *GetVectorCount*);
- *GetVectorWeight* — вес вектора.

```c++
class NEOML_API IProblem : virtual public IObject {
public:
	virtual ~IProblem() = 0;

	// Получить количество классов
	virtual int GetClassCount() const = 0;

	// Получить количество признаков
	virtual int GetFeatureCount() const = 0;

	// Является ли признак дискретным.
	virtual bool IsDiscreteFeature( int index ) const = 0;

	// Количество векторов в наборе данных.
	virtual int GetVectorCount() const = 0;

	// Получить номер класса для вектора [0, GetClassCount())
	virtual int GetClass( int index ) const = 0;

	// Получить вектор набора.
	virtual CSparseFloatVectorDesc GetVector( int index ) const = 0;

	// Получить все вектора набора.
	virtual CSparseFloatMatrixDesc GetMatrix() const = 0;

	// Получить вес вектора.
	virtual double GetVectorWeight( int index ) const = 0;
};
```

### Пример реализации

В библиотеке доступна одна простая реализация интерфейса `IProblem` — класс `CMemoryProblem`. Он хранит все данные в памяти.


## Для регрессии

Для обучения модели, решающей задачу регрессии, данные передаются в виде объекта, реализующего интерфейс `IRegressionProblem` (в случае функции, возвращающей скаляр) или `IMultivariateRegressionProblem` (в случае функции, возвращающей вектор). Оба интерфейса наследуются от базового интерфейса `IBaseRegressionProblem`.

Объект должен содержать всю информацию для обучения модели. Основные данные — это набор векторов, каждый из которых содержит значения нескольких признаков для одного объекта:

- *GetFeatureCount* — число признаков, которыми характеризуются объекты (т.е. длина одного вектора входных данных);
- *GetVectorCount* — число векторов в наборе, т.е. различных объектов, на которых вычисляются значения функции;
- *GetVector* — вектор с заданным порядковым номером;
- *GetMatrix* — все вектора набора в виде матрицы (размером *GetFeatureCount* * *GetVectorCount*);
- *GetVectorWeight* — вес вектора;
- *GetValue* — значение функции на векторе с заданным номером (скаляр в случае `IRegressionProblem`, вектор в случае `IMultivariateRegressionProblem`);
- *GetValueSize* — (только для `IMultivariateRegressionProblem`) длина вектора-значения функции.

```c++
class IBaseRegressionProblem : virtual public IObject {
public:
	// Получить количество признаков.
	virtual int GetFeatureCount() const = 0;

	// Количество векторов в наборе данных.
	virtual int GetVectorCount() const = 0;

	// Получить вектор из набора.
	virtual CSparseFloatVectorDesc GetVector( int index ) const = 0;

	// Получить все вектора набора.
	virtual CSparseFloatMatrixDesc GetMatrix() const = 0;

	// Получить вес вектора.
	virtual double GetVectorWeight( int index ) const = 0;
};

// Набор данных для задачи регрессии скалярнозначной функции.
class IRegressionProblem : public IBaseRegressionProblem {
public:
	// Получить значение для вектора.
	virtual double GetValue( int index ) const = 0;
};

// Набор данных для задачи регрессии векторнозначной функции.
class IMultivariateRegressionProblem : public IBaseRegressionProblem {
public:
	// Получить размерность вектора значений.
	virtual int GetValueSize() const = 0;
	// Получить значение для вектора.
	virtual CFloatVector GetValue( int index ) const = 0;
};
```
