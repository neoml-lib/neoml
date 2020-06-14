# Интерфейсы обученных моделей

<!-- TOC -->

- [Интерфейсы обученных моделей](#интерфейсы-обученных-моделей)
    - [Для классификации](#для-классификации)
    - [Для регрессии](#для-регрессии)
    - [Загрузка/сохранение модели](#загрузкасохранение-модели)
        - [Пример сохранения](#пример-сохранения)
        - [Пример загрузки](#пример-загрузки)

<!-- /TOC -->

Обученные модели для классификации или регрессии реализуют общие интерфейсы `IModel`, `IRegressionModel`, `IMultivariateRegressionModel`. Эти интерфейсы предоставляют методы, с помощью которых можно использовать модель, сохранять её в файл и загружать обратно.

## Для классификации

Модели классификаторов поддерживают интерфейс `IModel`. Он предоставляет методы `Classify` для классификации данных и `Serialize` для загрузки и сохранения модели.

```c++
class NEOML_API IModel : virtual public IObject {
public:
	virtual ~IModel() = 0;

	// Возвращает количество классов классификатора.
	virtual int GetClassCount() const = 0;

	// Классификация данных. Если не удалось классифицировать данные, то возвращает false.
	virtual bool Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const = 0;

	// Сериализация.
	virtual void Serialize( CArchive& archive ) = 0;
};
```

## Для регрессии

Модели для решения задачи регрессии представлены интерфейсами `IRegressionModel` (для функции, возвращающей скаляр) и `IMultivariateRegressionModel` (для функции, возвращающей вектор). Они предоставляют методы `Predict` для предсказания значений функции и `Serialize` для сохранения и загрузки модели.

```c++
// Модель задачи регрессии скалярнозначной функции.
class IRegressionModel : virtual public IObject {
public:
	// Предсказать значения для вектора.
	virtual double Predict( const CSparseFloatVector& data ) const = 0;
	virtual double Predict( const CFloatVector& data ) const = 0;
	virtual double Predict( const CSparseFloatVectorDesc& desc ) const = 0;

	// Сериализация.
	virtual void Serialize( CArchive& archive ) = 0;
};

// Модель задачи регрессии векторнозначной функции.
class IMultivariateRegressionModel : virtual public IObject {
public:
	// Предсказать значения для вектора.
	virtual CFloatVector MultivariatePredict( const CSparseFloatVector& data ) const = 0;
	virtual CFloatVector MultivariatePredict( const CFloatVector& data ) const = 0;

	// Сериализация.
	virtual void Serialize( CArchive& archive ) = 0;
};
```

## Загрузка/сохранение модели

Для загрузки и сохранения моделей используйте метод `Serialize` и класс `CArchive`.

### Пример сохранения

```c++
void StoreModel( CArchive& archive, IModel& model )
{
	CString modelName = GetModelName( &model );
	archive << modelName;
	model.Serialize( archive );
}
```

### Пример загрузки

```c++
CPtr<IModel> LoadModel( CArchive& archive )
{
	CString name;
	archive >> name;
	CPtr<IModel> result = CreateModel<IModel>( name );
	result->Serialize( archive );
	return result;
}
```
