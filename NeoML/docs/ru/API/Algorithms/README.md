# Различные алгоритмы

<!-- TOC -->

- [Различные алгоритмы](#различные-алгоритмы)
	- [Метод дифференциальной эволюции](#метод-дифференциальной-эволюции)
		- [Выбор начальной популяции](#выбор-начальной-популяции)
		- [Скрещивание/мутация](#скрещиваниемутация)
		- [Отбор](#отбор)
		- [Критерии остановки](#критерии-остановки)
		- [Оптимизируемая функция](#оптимизируемая-функция)
		- [Пример запуска](#пример-запуска)
	- [Генераторы гипотез](#генераторы-гипотез)
		- [Генератор путей в ориентированном ациклическом графе CGraphGenerator](#генератор-путей-в-ориентированном-ациклическом-графе-cgraphgenerator)
		- [Генератор паросочетаний CMatchingGenerator](#генератор-паросочетаний-cmatchinggenerator)
		- [Генератор последовательностей элементов фиксированной длины CSimpleGenerator](#генератор-последовательностей-элементов-фиксированной-длины-csimplegenerator)

<!-- TOC -->

Этот раздел описывает некоторые дополнительные алгоритмы, реализованные в библиотеке **NeoML**.

## Метод дифференциальной эволюции

Метод оптимизации, предназначенный для нахождения глобального минимума (или максимума) не дифференцируемых, нелинейных, мультимодальных функций от многих переменных F(x1, x2, ..., xn).

Метод работает следующим образом: из текущей популяции (набор параметров функции) с помощью операций скрещивания, мутации и отбора формируется следующая популяция, на элементах которой значения оптимизируемой функции лучше, и так до тех пор, пока не выполнены критерии останова.

### Выбор начальной популяции

Начальная популяция (x11, x12, ..., x1n) (x21, x22, ..., x2n) ... (xk1, xk2, ..., xkn) выбирается случайным образом.

### Скрещивание/мутация

Текущее поколение:

(x11, x12, ..., x1n) (x21, x22, ..., x2n) ... (xk1, xk2, ..., xkn).

Следующее поколение:

(y11, y12, ..., y1n) (y21, y22, ..., y2n) ... (yk1, yk2, ..., ykn).

Переход:

yij = xij | ( cij + fl * (aij - bij) ) : a, b, c - случайные вектора текущей популяции.

### Отбор

Элемент из текущей популяции заменяется на полученный в результате скрещивания и мутации, если значение оптимизируемой функции на нем "лучше":

```
yi = F(xi) < F(yi) ? xi : yi
```

### Критерии остановки

Алгоритм заканчивает работу при выполнении одного из условий:

- достигнуто ограничение числа итераций;
- минимум давно не обновлялся;
- произошло вырождение.

### Оптимизируемая функция

Функция для оптимизации описывается следующим интерфейсом:

```c++
class NEOML_API IFunctionEvaluation {
public:
	// размерность задачи
	virtual int NumberOfDimensions() const = 0;

	// "типы" параметров
	virtual const IParamTraits& GetParamTraits( int index ) const = 0;
	// тип возвращаемых значений
	virtual const IParamTraits& GetResultTraits() const  = 0;

	// Получить минимальное/максимальное значение вектора параметров
	virtual CFunctionParam GetMinConstraint( int index ) const = 0;
	virtual CFunctionParam GetMaxConstraint( int index ) const = 0;

	// Как минимум одна из функций Evaluate должна быть перегружена
	// Запуск с несколькими наборами параметров (по умолчанию просто вызывает запуск
	// с одним параметром несколько раз)
	virtual void Evaluate( const CArray<CFunctionParamVector>& params, CArray<CFunctionParam>& results );
	// Запуск с одним параметром (по умолчанию вызывает функцию с несколькими параметрами)
	virtual CFunctionParam Evaluate( const CFunctionParamVector& param );
};
```

### Пример запуска

Запустить алгоритм можно, например, так:

```c++
double fluctuation = 0.5; // коэффициент флуктуации.
double crossProbability = 0.5; // вероятность мутации.
const int populationSize = 100; // размер популяции.

CDifferentialEvolution evolution( func, fluctuation, crossProbability, populationSize );
evolution.SetMaxGenerationCount( 200 );
evolution.SetMaxNonGrowingBestValue( 10 );

evolution.RunOptimization();

evolution.GetOptimalVector();
```

## Генераторы гипотез

Описанные ниже алгоритмы генерируют наборы гипотез, которые затем можно использовать в различных сценариях, где требуется перебор вариантов.

### Генератор путей в ориентированном ациклическом графе CGraphGenerator

Данный алгоритм генерирует гипотезы в виде путей в ациклическом графе.

Исходные данные генерации представляют собой ориентированный ациклический граф, дуги которого имеют оценку. Дуги в узле отсортированы по убыванию качества суффикса пути до конца графа.

Генератор порождает пути из начальной точки графа в конечную в порядке убывания качества.

#### Пример использования

В данном примере создается генератор и генерируется первый путь.

```c++
CGraphGenerator<CGraph, CArc, int> generator( &graph );
generator.SetMaxStepsQueueSize( 1024 );

CArray<const CArc*> path;
generator.GetNextPath( path );
```

### Генератор паросочетаний CMatchingGenerator

Данный алгоритм генерирует гипотезы в виде паросочетаний в двудольном графе.

Паросочетания генерируются в порядке убывания качества.
На вход алгоритму подается граф, заданный матрицей штрафов за сочетания его вершин, а также штрафы за пропуск вершин правой и левой доли.

#### Пример использования

В данном примере создается генератор и генерируется оптимальное паросочетание.

```c++
CMatchingGenerator<CEdge, double> generator( leftSize, rightSize );

initializeMatrix( generator.PairMatrix() );
initializeLeftMissedElements( generator.MissedLeftElementPairs() );
initializeRightMissedElements( generator.MissedRightElementPairs() );

generator.Build();

CArray<CEdge> nextMatching;
generator.GetNextMatching( nextMatching );
```

### Генератор последовательностей элементов фиксированной длины CSimpleGenerator

Данный алгоритм генерирует гипотезы в виде последовательностей элементов фиксированной длины.

Исходные данные для генерации представляют собой множество упорядоченных по убыванию качества массивов альтернативных вариантов элементов.
На каждом шаге генератор порождает множество вариантов элементов таким образом, что суммарное качество порождаемых множеств не возрастает.

#### Пример использования

В данном примере генерируются последовательности целых чисел длиной 5.

```c++
const int NumberOfElement = 5;
const int NumberOfVariant = 3;

class CIntElement {
public:
	typedef int Quality;

	CIntElement() : value( 0 ) {}
	explicit CIntElement( int _value ) : value( _value ) {}

	int VariantQuality() const { return value; }

private:
	int value;
};

class CIntSimpleGenerator : public CSimpleGenerator<CIntElement> {
public:
	CIntSimpleGenerator() : 
		CSimpleGenerator<CIntElement>( 0, -10000 )
	{
		Variants.SetSize( NumberOfElement );
		for( int i = NumberOfVariant; i > 0; i-- ) {
			for( int j = 0; j < NumberOfElement; j++ ) {
				Variants[j].Add( CIntElement( i ) );
			}
		}
	}
};

CIntSimpleGenerator generator;

CArray<CIntElement> next;
generator.GetNextSet( next );
generator.GetNextSet( next );

```
