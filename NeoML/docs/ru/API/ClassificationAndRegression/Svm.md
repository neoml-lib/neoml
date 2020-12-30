# Машина опорных векторов CSvmClassifier

- [Машина опорных векторов CSvmClassifier](#машина-опорных-векторов-csvmclassifier)
	- [Параметры построения модели](#параметры-построения-модели)
	- [Модель](#модель)
	- [Пример](#пример)

Метод опорных векторов переводит исходные вектора в пространство более высокой размерности и ищет в нем разделяющую гиперплоскость с максимальным зазором.

В **NeoML** алгоритм реализован классом `CSvmClassifier`. Он предоставляет метод `Train` для обучения модели бинарной классификации.

## Параметры построения модели

Параметры реализованы структурой `CSvmClassifier::CParams`.

- *KernelType* — тип используемого ядра;
- *ErrorWeight* — вес "ошибки" относительно регуляризатора;
- *MaxIterations* — ограничение на число итераций;
- *Degree* — степень гауссова ядра;
- *Gamma* — коэффициент ядра (используется для `KT_Poly`, `KT_RBF`, `KT_Sigmoid`);
- *Coeff0* — независимый член в функции ядра (используется для `KT_Poly`, `KT_Sigmoid`);
- *Tolerance* — точность нахождения решения, критерий останова;
- *ThreadCount* — количество потоков, используемых при работе алгоритма.

## Модель

Построенная модель может описываться интерфейсом [`ILinearBinaryModel`](Linear.md#для-классификации) при использовании ядра `KT_Linear` и интерфейсом `ISvmBinaryModel` в остальных случаях.

```c++
// интерфейс SVM binary классификатора
class ISvmBinaryModel : public IModel {
public:
	virtual ~ISvmBinaryModel();

	// получить ядро
	virtual CSvmKernel::TKernelType GetKernelType() const = 0;

	// получить опорные вектора
	virtual CSparseFloatMatrix GetVectors() const = 0;

	// получить коэффициенты при опорных векторах
	virtual const CArray<double>& GetAlphas() const = 0;

	// получить свободный член
	virtual double GetFreeTerm() const = 0;

	// Сериализация.
	virtual void Serialize( CArchive& ) = 0;
};
```

## Пример

Ниже представлен простой пример обучения модели методом машины опорных векторов.

```c++
CPtr<Model> buildModel( IProblem* data )
{
	CSvmClassifier::CParams params( CSvmKernel::KT_RBF );
	CSvmClassifier builder( params );
	return builder.Train( *data );
}
```