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
			- [Пример использования](#пример-использования)
		- [Генератор паросочетаний CMatchingGenerator](#генератор-паросочетаний-cmatchinggenerator)
			- [Пример использования](#пример-использования-1)
		- [Генератор последовательностей элементов фиксированной длины CSimpleGenerator](#генератор-последовательностей-элементов-фиксированной-длины-csimplegenerator)
			- [Пример использования](#пример-использования-2)
	- [Кодирование подслов для языковых моделей](#кодирование-подслов-для-языковых-моделей)
		- [Кодирование пар байтов BPE](#кодирование-пар-байтов-bpe)

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
// Функция оценивающая параметры CSvm
// Из-за того что дифференциальная эволющия подбирает каждый параметр линейно на промежутке [min;max]
// Поэтому для некоторых параметров мы ищем их оптимальные логарифмы (по основанию 10)
class CSvmEvaluation : public IFunctionEvaluation {
private:
	// Подбираемые параметры CSvm
	enum TSvmParam {
		SP_KernelType, // Тип ядра CSvm (enum интерпретируемый как int)
		SP_LogErrorWeight, // Логарифм параметра ErrorWeight
		SP_MaxIterations, // Максимальное число итераций
		SP_Degree, // Параметр Degree
		SP_LogGamma, // Логарифм параметра Gamma
		SP_LogCoeff0, // Логарифм параметра Coeff0
		SP_LogTolerance, // Логарифм параметра Tolerance

		SP_Count // Размер вектора параметров
	};
public:
	// Принимает на вход данные и количество частей, используемых при кросс-валидации
	explicit CSvmEvaluation( const IProblem& problem, int cvFolds ) :
		problem( &problem ), cvFolds( cvFolds ) {}

	// IFunctionEvaluation interface 

	// Число элементов в векторе параметров
	int NumberOfDimensions() const override { return static_cast<int>( SP_Count ); }

	// Тип каждого параметра в векторе
	const IParamTraits& GetParamTraits( int index ) const override
	{
		switch( static_cast<TSvmParam>( index ) ) {
			case SP_KernelType:
			case SP_MaxIterations:
			case SP_Degree:
				return CIntTraits::GetInstance();
			case SP_LogErrorWeight:
			case SP_LogGamma:
			case SP_LogCoeff0:
			case SP_LogTolerance:
				return CDoubleTraits::GetInstance();
			case SP_Count:
			default:
				NeoAssert( false );
		}
		return CIntTraits::GetInstance();
	}

	// Тип оптимизируемого значения 
	const IParamTraits& GetResultTraits() const override { return CDoubleTraits::GetInstance(); }

	// Минимальные значения каждого из элементов
	CFunctionParam GetMinConstraint( int index ) const override
	{
		switch( static_cast<TSvmParam>( index ) ) {
			case SP_KernelType:
				return CIntTraits::GetInstance().Box( static_cast<int>( CSvmKernel::KT_Linear ) );
			case SP_LogErrorWeight:
				return CDoubleTraits::GetInstance().Box( -3. );
			case SP_MaxIterations:
				return CIntTraits::GetInstance().Box( 10 );
			case SP_Degree:
				return CIntTraits::GetInstance().Box( 1 );
			case SP_LogGamma:
				return CDoubleTraits::GetInstance().Box( -3. );
			case SP_LogCoeff0:
				return CDoubleTraits::GetInstance().Box( -3. );
			case SP_LogTolerance:
				return CDoubleTraits::GetInstance().Box( -4 );
			default:
				NeoAssert( false );
		}
		return CDoubleTraits::GetInstance().Box( 1 );
	}

	// Максимальные значения каждого из элементов
	CFunctionParam GetMaxConstraint( int index ) const override
	{
		switch( static_cast<TSvmParam>( index ) ) {
			case SP_KernelType:
				return CIntTraits::GetInstance().Box( static_cast<int>( CSvmKernel::KT_Sigmoid ) );
			case SP_LogErrorWeight:
				return CDoubleTraits::GetInstance().Box( 3. );
			case SP_MaxIterations:
				return CIntTraits::GetInstance().Box( 1000 );
			case SP_Degree:
				return CIntTraits::GetInstance().Box( 5 );
			case SP_LogGamma:
				return CDoubleTraits::GetInstance().Box( 3 );
			case SP_LogCoeff0:
				return CDoubleTraits::GetInstance().Box( 3 );
			case SP_LogTolerance:
				return CDoubleTraits::GetInstance().Box( -1 );
			default:
				NeoAssert( false );
		}
		return CDoubleTraits::GetInstance().Box( 1 );
	}

	// Оценка одного вектора параметров
	// В нашем случае это средний результат кросс-валидации CSvm с этими параметрами
	// на данных, переданных в конструкторе
	CFunctionParam Evaluate( const CFunctionParamVector& param ) override
	{
		// Некоторые параметры использовать как показатель степени!
		CSvm::CParams svmParams(
			static_cast<CSvmKernel::TKernelType>( CIntTraits::GetInstance().Unbox( param[SP_KernelType] ) ),
			::pow( 10., CDoubleTraits::GetInstance().Unbox( param[SP_LogErrorWeight] ) ),
			CIntTraits::GetInstance().Unbox( param[SP_MaxIterations] ),
			CIntTraits::GetInstance().Unbox( param[SP_Degree] ),
			::pow( 10., CDoubleTraits::GetInstance().Unbox( param[SP_LogGamma] ) ),
			::pow( 10., CDoubleTraits::GetInstance().Unbox( param[SP_LogCoeff0] ) ),
			::pow( 10., CDoubleTraits::GetInstance().Unbox( param[SP_LogTolerance] ) ),
			true,
			OmpGetMaxThreadCount(),
			MM_OneVsOne
		);

		CSvm svm( svmParams );
		CCrossValidation cv( svm, problem );
		CCrossValidationResult cvResult;
		cv.Execute( cvFolds, AccuracyScore, cvResult, true );

		double total = 0;
		for( int i = 0; i < cvResult.Success.Size(); ++i ) {
			total += cvResult.Success[i];
		}
		// В данной задаче мы пытаемся максимизировать accuracy.
		// А дифф. эволюция пытается найти минимум целевого значения.
		// Потому используем -accuracy как оптимизируемое значение.
		return CDoubleTraits::GetInstance().Box( -total / cvResult.Success.Size() );
	}

private:
	CPtr<const IProblem> problem;
	int cvFolds;
};

double fluctuation = 0.5; // коэффициент флуктуации.
double crossProbability = 0.5; // вероятность мутации.
const int populationSize = 20; // размер популяции.

CSvmEvaluation svmEval( *problem, 5 );
CDifferentialEvolution evolution( svmEval, fluctuation, crossProbability, populationSize );
evolution.SetMaxGenerationCount( 100 );
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

## Кодирование подслов для языковых моделей

Токенизация и кодирование подслов — подход, имеющий преимущества перед подходами, в которых токенами служат отдельные символы или целые слова.

```c++
// Интерфейс механизма кодирования подслов.
class NEOML_API ISubwordEncoder : virtual public IObject {
public:
	virtual ~ISubwordEncoder() override = default;

	// Кодирование слова в виде последовательности идентификаторов токенов вместе с длинами токенов.
	// Значение идентификатора каждого токена лежит в диапазоне [0, ... , Size() - 1].
	// Для кодирования слова, содержащего символы за пределами ASCII, необходимо предварительно кодировать его в UTF-8 и
	// передать соответствующий экземпляр CString.
	// В этом случае массив 'tokenLengths' будет содержать длины токенов согласно исходной версии слова.
	virtual void Encode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const = 0;
	
	// Декодирует последовательность идентификаторов токенов в последовательность слов.
	virtual void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const = 0;

	// Возвращает кол-во токенов.
	virtual int Size() const = 0;

	// Получение отображений слово <-> идентификатор, как их производит кодировщик
	virtual void GetIdToTokenMapping( CMap<int, CString>& ) const = 0;
	virtual void GetTokenToIdMapping( CMap<CString, int>& ) const = 0;

	struct CParams {
		// Специальный токен конца слова для каждого кодируемого слова.
		CString UseEndOfWordToken = "";
		// Специальный токен начала слова для каждого кодируемого слова.
		CString UseStartOfWordToken = "";
		// Работать со строками как с массивами байт.
		// Позволяет сократить и полностью избежать применение неизвестных токенов, т.к. всего различных байт 256 и, 
		// в отличие от букв юникода, их реально перечислить в стартовом словаре
		bool UseRawBytes = false;
		// Идентификатор "неизвестного токена".
		// Прочие токены нумеруются непрерывно с 'UnknownTokenId' + 1.
		// Номера с 0 по 'UnknownTokenId' не используются токенизатором и могут быть задействованы пользователем для служебных символов. 
		int UnknownTokenId = 0
	};

	// Возвращают флаги механизма кодирования.
	virtual bool UseEndOfWordToken() const = 0;
	virtual bool UseStartOfWordToken() const = 0;
	virtual bool UseRawBytes() const = 0;
	virtual int UnknownTokenId() const = 0;
};
```

Некоторые алгоритмы кодирования могут быть ускорены путем использования кеширования для вызовов `Encode`.
Большое кол-во вызовов `Encode` обычно случается в процессе обучения языковой модели.

Для этих целей был добавлен дополнительный интерфейс:

```c++
// Механизм кодирования подслов, поддерживающий кеширование результатов вызовов метода `Encode`.
class NEOML_API ISubwordEncoderWithCache : public ISubwordEncoder {
public:
	virtual void Encode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const override final;

	// Метод устанавливает период очистки кеша.
	// Кеш используется для ускорения работы метода Encode.
	// Результат вызова Encode добавляется в кеш и впоследствии удаляется,
	// если среди последующих ~cachePeriod вызовов метода Encode
	// не будет вызова с таким же аргументом.
	// Увеличение cachePeriod приводит к увеличению потребления памяти.
	// Для отключения механизма кеширования необходимо передать -1 в качестве аргумента.
	// Значение cachePeriod равное 0 недопустимо.
	void SetCachePeriod( int cachePeriod ) const { cache.SetCachePeriod( cachePeriod ); }
```

### Кодирование пар байтов BPE

Популярный алгоритм кодирования подслов.

```c++
class NEOML_API IBytePairEncoder : public ISubwordEncoderWithCache {
public:
	// Словарь задается списком уникальных токенов, упорядоченных по порядку добавления в словарь при обучении (т.е. по убыванию частоты). 
	// Идентификатор при кодировании равен позиции в этом массиве + GetUnknownTokenId() + 1
	using CBPEDictionary = CArray<CString>;

	// Загрузить обученный внешними инструментами механизм
	virtual void Initialize( const CBPEDictionary& tokens, const CParams& ) = 0;
};
```

**Unigram** language model ([Kudo.](https://arxiv.org/abs/1804.10959)) - алгоритм, альтернативный BPE.

```c++
class NEOML_API IUnigramEncoder : public ISubwordEncoderWithCache {
public:
	// Элемент словаря
	struct CSubtoken {
		CSubtoken() = default;
		CSubtoken( CString text, double score );
		void Serialize( CArchive& archive );

		CString Text;
		double Score = 0.0;
	};

	// Словарь задается списком уникальных токенов
	using CUnigramDictionary = CArray<CSubtoken>;

	// Загрузить словарь, обученный внешними инструментами
	virtual void Initialize( const CUnigramDictionary& tokens, const CParams& ) = 0;

	// Получить словарь
	virtual void GetDictionary( CUnigramDictionary& tokens ) const = 0;
};
```

Для обучения механизма кодирования, удовлетворяющего интерфейсу `IBytePairEncoder`, нужно воспользоваться классом `CSubwordEncoderTrainer`.

```c++
// Класс, осуществляющий обучение механизма субтокенного кодирования.
class NEOML_API CSubwordEncoderTrainer {
public:
    // Обучение IBytePairEncoder
	enum class TAlgorithm {
		BPE
		// Обучение Unigram в разработке
	};

	// Варианты обработки границ: автоматически добавляет спец. символ в начало и\или конец слова при обучении
	enum class TBorderHandling {
		EndOfWord,
		BeginOfWord,
		SentencePiece,
		BeginAndEndOfWord,
		None
	};

	// Ограничение первичного (односимвольного) словаря. Решает проблему множественных редких символов юникода, чье наличие в итоговом словаре нежелательно.
	enum class TVocabPruning {
		// Односимвольные токены добавляются по убыванию частоты до достижения заданного покрытия. По умолчанию 1, т.е. добавятся все.
		Coverage,
		// Символом считается байт, первичный словарь имеет размер 255 и не требует сокращения.
		ByteBPE
	};

	CSubwordEncoderTrainer( int vocabSize, TAlgorithm, TBorderHandling, TVocabPruning = TVocabPruning::Coverage );

	// Установить порог для TVocabPruning = TVocabPruning::Coverage
	void SetCharacterCoverage( double value );
	// Установить список односимвольных токенов, которые будут добавлены вне зависимости от ограничения по покрытию
	void SetMandatoryChars( const CArray<CString>& );
	// Установить сдвиг словаря. Нумерация токенов непрерывна и начинается с UnknownTokenId. По умолчанию 0.
	void SetUnknownTokenId( int value );

	// Обучает и возвращает полностью обученный кодировщик.
	CPtr<ISubwordEncoder> Train( const CWordDictionary& frequencyDict );
```

Итоговая последовательность шагов для обучения кодировщика:

1. Создать словарь с частотами на основе текстового корпуса, используя экземпляр класса `CWordDictionary`.
2. Создать экземпляр класса `CSubwordEncoderTrainer` и настроить желаемые параметры.
3. Вызвать метод `Train` у экземпляра `CBytePairEncoderTrainer`.

В целях инференса внешних моделей `IBytePairEncoder` и `IUnigramEncoder` предоставляют прямой метод загрузки словаря `Initialize`. Для BPE словарь задаётся с помощью массива, где токены располагаются в порядке обучения словаря, т.е. по убыванию частоты. Словарь, созданный не алгоритмами NeoML, должен соответствовать следующим требованиям энкодера:
1. Каждый токен, кроме букв, должен быть конкатенацией двух меньших токенов.
2. End-Of-Word может располагаться только на окончании токенов. Start-Of-Word можен располагаться только в начале токенов.
3. End-Of-Word и Start-Of-Word должны содержаться в словаре как отдельные токены (вообще говоря, это следует из вышеизложенных правил).
   
Для Unigram требуется загрузить пары токен-score.
