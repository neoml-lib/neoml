# Нейронные сети

<!-- TOC -->

- [Нейронные сети](#нейронные-сети)
    - [Выбор вычислительного движка](#выбор-вычислительного-движка)
    - [Блобы данных](#блобы-данных)
    - [Принципы нейронных сетей](#принципы-нейронных-сетей)
        - [Концепция слоя](#концепция-слоя)
        - [Сеть CDnn](#сеть-cdnn)
    - [Обучение сети](#обучение-сети)
        - [Инициализация весов](#инициализация-весов)
        - [Методы оптимизации](#методы-оптимизации)
        - [Запуск с обучением](#запуск-с-обучением)
        - [Запуск без обучения](#запуск-без-обучения)
    - [Сериализация](#сериализация)
        - [Пример сохранения сети](#пример-сохранения-сети)
    - [Использование сети](#использование-сети)
    - [Список слоёв](#список-слоёв)

<!-- /TOC -->

## Выбор вычислительного движка

Перед работой с нейронными сетями необходимо определиться, какое устройство будет использоваться для вычислений (CPU или GPU). Создайте соответствующий [вычислительный движок](MathEngine.md) и передавайте указатель на него при создании сети и слоёв для неё.

## Блобы данных

Любые данные, используемые при работе с сетью (входы, выходы, обучаемые параметры) хранятся в [блобах](DnnBlob.md). Блоб представляет собой 7-мерный массив, каждая размерность которого имеет определенное значение:

- `BatchLength` - "временная" шкала, используемая для обозначения последовательностей данных; обычно применяется в рекуррентных сетях;
- `BatchWidth` - батч, используется для одновременной передачи нескольких не связанных между собой объектов;
- `ListSize` - размерность, используемая для обозначения того, что объекты связаны между собой (например, это могут быть пиксели, извлеченные из одного изображения), но при этом не являются последовательностью;
- `Height` - высота, используется при работе с матрицами или изображениями;
- `Width` - ширина, используется при работе с матрицами или изображениями;
- `Depth` - глубина, используется при работе с трехмерными изображениями;
- `Channels` - каналы, используется при работе с многоканальными изображениями, а также при работе с одномерными векторами.

Поддерживаются два типа данных: с плавающей точкой (`CT_Float`) и целочисленный (`CT_Int`). В обоих случаях используются 32-битные типы данных. Если где-либо в этой документации описание блоба не содержит явного указания типа данных, то подразумеваются данные с плавающей точкой.

## Принципы нейронных сетей

### Концепция слоя

[Слой](BaseLayer.md) - это элемент сети, выполняющий некоторую операцию. Операцией в этом случае может быть что угодно, от изменения формы входных данных или вычисления простой математической функции до свёртки или LSTM ([Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)).

Если выполняемая операция подразумевает наличие входных данных, то они будут взяты из входов слоя. Каждый вход слоя содержит один блоб с данными, поэтому у слоев, выполняющих операцию над несколькими блобами данных, будет несколько входов. Перед работой каждый вход слоя необходимо [присоединить](BaseLayer.md#присоединение-к-другим-слоям) к какому-либо выходу другого слоя.

Если выполняемая операция подразумевает вычисление результатов, которые будут затем использоваться другими слоями, то они будет переданы в выходы слоя. Каждый выход содержит один блоб с результатами, поэтому в зависимости от типа операции слой может иметь несколько выходов. К одному выходу слоя могут быть присоединены *несколько* входов других слоёв, однако не допускается ситуация, когда к выходу слоя ничего не присоединено.

Также у слоя могут быть настройки, задаваемые пользователем перед вычислениями, и обучаемые параметры, оптимизируемые во время обучения сети.

Для возможности идентификации слоя в сети используются [имена](BaseLayer.md#имя-слоя), задаваемые до добавления слоя в сеть.

Полный список слоёв со ссылками на подробные их описания см. [ниже](#список-слоёв).

### Сеть CDnn

Нейронная сеть реализована при помощи класса [CDnn](Dnn.md) и представляет собой направленный граф, вершины которого обозначают слои, а рёбра обозначают передачи данных от выходов одних слоёв на входы других.

Для включения слоя в сеть, его необходимо туда [добавить](Dnn.md#добавление-слоя), предварительно установив ему уникальное, в рамках этой сети, [имя](BaseLayer.md#имя-слоя). Слой не может одновременно использоваться в нескольких сетях.

Для передачи данных в сеть используются [слои-источники](IOLayers/SourceLayer.md), не имеющие входов и передающие заранее установленный пользователем блоб данных в свой единственный выход.

Для получения данных после работы сети используются [специальные слои](IOLayers/SinkLayer.md), не имеющие выходов, из которых в дальнейшем можно извлечь данные.

После включения всех слоёв сеть и установки всех соединений между ними можно готовить сеть к обучению.

## Обучение сети

Для обучения сети вам понадобится:

* слой, вычисляющий оптимизируемую [функцию потерь](LossLayers/README.md) (или несколько таких слоёв);
* еще несколько слоёв-источников, в которых будут передаваться в сеть правильных меток и весов объектов;
* установить параметры инициализации весов и используемый при обучении метод оптимизации.

### Инициализация весов

Перед первым шагом обучения веса инициализируются при помощи специального объекта `CDnnInitializer`, который имеет 2 реализации:

- `CDnnUniformInitializer` - класс, генерирующий веса из равномерного распределения на отрезке, границы которого устанавливаются методами `GetLowerBound` и `GetUpperBound`;
- `CDnnXavierInitializer` - класс, генерирующий веса из нормального распределения `N(0, 1/n)`, где `n` - число входных нейронов у слоя.

Для задания инициализации необходимо создать объект нужного класса и передать его в сеть при помощи метода [`CDnn::SetInitializer`](Dnn.md#инициализация-весов). По умолчанию используется инициализация `Xavier`.

Инициализация задается единожды на всю сеть и используется для задания начальных значений всем обучаемым параметрам, кроме векторов свободных членов. Вектора свободных членов всегда инициализируются нулями.

### Методы оптимизации

Методы оптимизации задают правила обновления весов во время обучения. За это отвечает специальный класс `CDnnSolver`, который имеет 4 реализации:

- `CDnnSimpleGradientSolver` - градиентный спуск с моментом (см. [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum))
- `CDnnAdaptiveGradientSolver` - градиентный спуск с адаптивным моментом (см. [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam))
- `CDnnNesterovGradientSolver` - Adam с моментом Нестерова (см. [Nadam](http://cs229.stanford.edu/proj2015/054_report.pdf))
- `CDnnLambGradientSolver` - [LAMB](https://arxiv.org/pdf/1904.00962.pdf).

Для задания метода оптимизации необходимо создать объект нужного класса и передать его в сеть при помощи метода [`CDnn::SetSolver`](Dnn.md#solver).

Также в рамках метода оптимизации задаются:

- скорость сходимости (`CDnnSolver::SetLearningRate`);
- коэффициенты регуляризации (`CDnnSolver::SetL2Regularization` и `CDnnSolver::SetL1Regularization`).

### Запуск с обучением

После установки инициализации и метода оптимизации можно приступать к обучению. Для этого необходимо установить блобы с данными всем слоям-источникам и вызвать метод `CDnn::RunAndLearnOnce`.

Этот метод внутри состоит из 3 шагов:

1. `Reshape` - подсчёт размеров и аллокация блобов выходов у всех слоёв на основе размеров блобов в слоях-источниках;
2. `RunOnce` - вычисление всех операций в сети на данных из слоёв-источников;
3. `BackwardAndLearnOnce` - вычисление градиентов функции ошибки для обучаемых параметров всех слоёв и обновление этих обучаемых параметров.

Сам процесс обучения представляет собой многократное заполнение блобов слоёв источников разными данными и запуском `CDnn::RunAndLearnOnce`.

### Запуск без обучения

Во время обучения зачастую необходимо получить ответ сети на некоторых данных, без обновления параметров. Например, для валидации. Для этого используется метод `CDnn::RunOnce` который отличается от `CDnn::RunAndLearnOnce` тем, что не содержит шага подсчета градиентов и обновления параметров. Этот же метод используется для работы с сетью после обучения.

## Сериализация

Для сериализации сетей используются два класса:

- `CArchiveFile` - файл, используемый при сериализации;
- `CArchive` - архив, сериализующий в `CArchiveFile`.

Направление сериализации определяются флагами, с которыми создаётся файл и архив:

* для сохранения сети в файл нужно создать файл с флагом `CArchive::store` и создать над ним архив с флагом `CArchive::SD_Storing`;
* для чтения сети из файла нужно использовать флаги `CArchive::load` и `CArchive::SD_Loading` соответственно.

После создания архива сериализация производится методом [`CDnn::Serialize`](Dnn.md#сериализация), вне зависимости от направления.

Подробнее о классах для сериализации см. [тут](../Common/README.md#сериализация).

### Пример сохранения сети

```c++
CRandom random( 0x123 );
CDnn net( random, GetDefaultCpuMathEngine() );

/*
... Построение сети и её обучение ...
*/

CArchiveFile file( "my_net.archive", CArchive::store );
CArchive archive( &file, CArchive::SD_Storing );
archive.Serialize( net );
archive.Close();
file.Close();
```

## Использование сети

```c++
// Вычислительный движок, работающий на видеокарте
// и использующий не более чем 1 гигабайт видеопамяти.
IMathEngine* gpuMathEngine = CreateGpuMathEngine( 1024 * 1024 * 1024, GetFmlExceptionHandler() );

{
    CRandom random( 0x123 );
    CDnn net( random, *gpuMathEngine );

    // Загружаем сеть.
    {
      CArchiveFile file( "my_net.archive", CArchive::store );
      CArchive archive( &file, CArchive::SD_Storing );
      archive.Serialize( net );
      // file и archive будут закрыты в деструкторах объектов. 
    }

    // Блоб для одного RGB изображения 32x32
    CPtr<CDnnBlob> dataBlob = CDnnBlob::Create2DImageBlob( *gpuMathEngine, CT_Float, 1, 1, 32, 32, 3 );

    dataBlob->Fill( 0.5f ); // Для примера, заполним константой.

    // Получим из сети указатели на слои-источники и выходы
    CPtr<CSourceLayer> src = CheckCast<CSourceLayer>( net.GetLayer( "source" ) );
    CPtr<CSinkLayer> sink = CheckCast<CSinkLayer>( net.GetLayer( "sink" ) );

    src->SetBlob( dataBlob ); // Задаем данные.
    net.RunOnce(); // Запускаем сеть.
    CPtr<CDnnBlob> resultBlob = sink->GetBlob(); // Получаем результаты.

    // Извлекаем результаты в массив.
    CArray<float> result;
    result.SetSize( resultBlob->GetDataSize() );
    resulBlob->CopyTo( result.GetPtr() );

    // Здесь можно проанализировать результаты.

    // Здесь будут разрушены все блобы и объект сети.
}

// Удаляем движок после удаления всех блобов.
delete gpuMathEngine;
```

## Список слоёв

- [CBaseLayer](BaseLayer.md) - базовый класс
- Слои для обмена данными с сетью:
  - [CSourceLayer](IOLayers/SourceLayer.md) - передача блобов с данными в сеть
  - [CSinkLayer](IOLayers/SinkLayer.md) - получение блобов с данными из сети
  - [CProblemSourceLayer](IOLayers/ProblemSourceLayer.md) - передача данных из [`IProblem`](../ClassificationAndRegression/Problems.md) в сеть
  - [CFullyConnectedSourceLayer](IOLayers/FullyConnectedSourceLayer.md) - передача данных из `IProblem` в сеть и домножение их на матрицу
- [CFullyConnectedLayer](FullyConnectedLayer.md) - полносвязный слой
- [Функции активации](ActivationLayers/README.md):
  - [CLinearLayer](ActivationLayers/LinearLayer.md) - функция активации вида `ax + b`
  - [CELULayer](ActivationLayers/ELULayer.md) - функция активации `ELU`
  - [CReLULayer](ActivationLayers/ReLULayer.md) - функция активации `ReLU`
  - [CLeakyReLULayer](ActivationLayers/LeakyReLULayer.md) - функция активации `LeakyReLU`
  - [CAbsLayer](ActivationLayers/AbsLayer.md) - функция активации `abs(x)`
  - [CSigmoidLayer](ActivationLayers/SigmoidLayer.md) - функция активации `sigmoid`
  - [CTanhLayer](ActivationLayers/TanhLayer.md) - функция активации `tanh`
  - [CHardTanhLayer](ActivationLayers/HardTanhLayer.md) - функция активации `HardTanh`
  - [CHardSigmoidLayer](ActivationLayers/HardSigmoidLayer.md) - функция активации `HardSigmoid`
  - [CPowerLayer](ActivationLayers/PowerLayer.md) - функция активации `pow(x, exp)`
  - [CHSwishLayer](ActivationLayers/HSwishLayer.md) - функция активации `h-swish`
  - [CGELULayer](ActivationLayers/GELULayer.md) - функция активации `x * sigmoid(1.702 * x)`
- Свертки:
  - [CConvLayer](ConvolutionLayers/ConvLayer.md) - двумерная свертка
    - [CRleConvLayer](ConvolutionLayers/RleConvLayer.md) - свертка двумерных изображений в формате RLE
  - [C3dConvLayer](ConvolutionLayers/3dConvLayer.md) - трехмерная свертка
  - [CTranposedConvLayer](ConvolutionLayers/TransposedConvLayer.md) - обратная двумерная свертка
  - [C3dTranposedConvLayer](ConvolutionLayers/3dTransposedConvLayer.md) - обратная трехмерная свертка
  - [CChannelwiseConvLayer](ConvolutionLayers/ChannelwiseConvLayer.md) - поканальная свертка
  - [CTimeConvLayer](ConvolutionLayers/TimeConvLayer.md) - свертка последовательностей "по времени"
- Пулинги:
  - [CMaxPoolingLayer](PoolingLayers/MaxPoolingLayer.md) - двумерный `Max Pooling`
  - [CMeanPoolingLayer](PoolingLayers/MeanPoolingLayer.md) - двумерный `Mean Pooling`
  - [C3dMaxPoolingLayer](PoolingLayers/3dMaxPoolingLayer.md) - трехмерный `Max Pooling`
  - [C3dMeanPoolingLayer](PoolingLayers/3dMeanPoolingLayer.md) - трехмерный `Mean Pooling`
  - [CGlobalMaxPoolingLayer](PoolingLayers/GlobalMaxPoolingLayer.md) - `Max Pooling` над объектами целиком
  - [CMaxOverTimePoolingLayer](PoolingLayers/MaxOverTimePoolingLayer.md) - `Max Pooling` над последовательностями "по времени"
- [CSoftmaxLayer](SoftmaxLayer.md) - вычисление функции `softmax`
- [CDropoutLayer](DropoutLayer.md) - реализация `dropout`
- [CBatchNormalizationLayer](BatchNormalizationLayer.md) - батч нормализация
- [CObjectNormalizationLayer](ObjectNormalizationLayer.md) - нормализация по объектам
- Поэлементные операции над блобами:
  - [CEltwiseSumLayer](EltwiseLayers/EltwiseSumLayer.md) - поэлементная сумма блобов
  - [CEltwiseMulLayer](EltwiseLayers/EltwiseMulLayer.md) - поэлементное произведение блобов
  - [CEltwiseMaxLayer](EltwiseLayers/EltwiseMaxLayer.md) - поэлементный максимум блобов
  - [CEltwiseNegMulLayer](EltwiseLayers/EltwiseNegMulLayer.md) - поэлементное произведение разности `1` и элементов первого блоба с элементами остальных блобов
- Вспомогательные операции:
  - [CTransformLayer](TransformLayer.md) - изменение формы блоба
  - [CTransposeLayer](TransposeLayer.md) - перестановка размерностей блоба
  - [CArgmaxLayer](ArgmaxLayer.md) - поиск максимумов вдоль некоторой размерности
  - [CImageResizeLayer](ImageResizeLayer.md) - изменение размера изображений в блобе
  - [CSubSequenceLayer](SubSequenceLayer.md) - выделение подпоследовательностей
  - [CDotProductLayer](DotProductLayer.md) - скалярное произведение объектов двух блобов
  - [CAddToObjectLayer](AddToObjectLayer.md) - прибавление содержимого одного входа ко всем объектам другого
  - [CMatrixMultiplicationLayer](MatrixMultiplicationLayer.md) - перемножение двух наборов матриц
  - Объединение блобов:
    - [CConcatChannelsLayer](ConcatLayers/ConcatChannelsLayer.md) - объединение блобов по каналам
    - [CConcatDepthLayer](ConcatLayers/ConcatDepthLayer.md) - объединение блобов по глубине
    - [CConcatWidthLayer](ConcatLayers/ConcatWidthLayer.md) - объединение блобов по ширине
    - [CConcatHeightLayer](ConcatLayers/ConcatHeightLayer.md) - объединение блобов по высоте
    - [CConcatBatchWidthLayer](ConcatLayers/ConcatBatchWidthLayer.md) - объединение блобов по `BatchWidth`
    - [CConcatObjectLayer](ConcatLayers/ConcatObjectLayer.md) - объединение объектов в блобах
  - Разделение блобов:
    - [CSplitChannelsLayer](SplitLayers/SplitChannelsLayer.md) - разделение блобов по каналам
    - [CSplitDepthLayer](SplitLayers/SplitDepthLayer.md) - разделение блобов по глубине
    - [CSplitWidthLayer](SplitLayers/SplitWidthLayer.md) - разделение блобов по ширине
    - [CSplitHeightLayer](SplitLayers/SplitHeightLayer.md) - разделение блобов по высоте
    - [CSplitBatchWidthLayer](SplitLayers/SplitBatchWidthLayer.md) - разделение блобов по `BatchWidth`
  - Работа со списками пикселей:
    - [CPixelToImageLayer](PixelToImageLayer.md) - построение изображений из списков пикселей
    - [CImageToPixelLayer](ImageToPixelLayer.md) - выделение списков пикселей из изображений
  - Повторение данных:
    - [CRepeatSequenceLayer](RepeatSequenceLayer.md) - повторение последовательностей несколько раз
    - [CUpsampling2DLayer](Upsampling2DLayer.md) - увеличение размеров двумерных изображений
  - [CReorgLayer](ReorgLayer.md) - слой, преобразующий многоканальные изображения в изображения меньшего размера, с большим числом каналов
- Функции потерь:
  - Бинарная классификация:
    - [CBinaryCrossEntropyLossLayer](LossLayers/BinaryCrossEntropyLossLayer.md) - перекрёстная энтропия
    - [CHingeLossLayer](LossLayers/HingeLossLayer.md) - функция `Hinge`
    - [CSquaredHingeLossLayer](LossLayers/SquaredHingeLossLayer.md) - модифицированная функция `SquaredHinge`
    - [CBinaryFocalLossLayer](LossLayers/BinaryFocalLossLayer.md) - функция `Focal` (модифицированная кросс-энтропия)
  - Многоклассовая классификация:
    - [CCrossEntropyLossLayer](LossLayers/CrossEntropyLossLayer.md) - перекрёстная энтропия
    - [CMultiHingeLossLayer](LossLayers/MultiHingeLossLayer.md) - функция `Hinge`
    - [CMultiSquaredHingeLossLayer](LossLayers/MultiSquaredHingeLossLayer.md) - модифицированная функция `SquaredHinge`
    - [CFocalLossLayer](LossLayers/FocalLossLayer.md) - функция `Focal` (модифицированная кросс-энтропия)
  - Регрессия:
    - [CEuclideanLossLayer](LossLayers/EuclideanLossLayer.md) - евклидово расстояние
  - Дополнительно:
    - [CCenterLossLayer](LossLayers/CenterLossLayer.md) - вспомогательная функция `Center`, штрафующая дисперсию внутри классов
- Работа с дискретными признаками:
  - [CMultichannelLookupLayer](DiscreteFeaturesLayers/MultichannelLookupLayer.md) - векторные представления дискретных признаков
  - [CAccumulativeLookupLayer](DiscreteFeaturesLayers/AccumulativeLookupLayer.md) - сумма векторных представлений дискретного признака
  - [CPositionalEmbeddingLayer](DiscreteFeaturesLayer/PositionalEmbeddingLayer.md) - векторные представления позиций в последовательности
  - [CEnumBinarizationLayer](DiscreteFeaturesLayers/EnumBinarizationLayer.md) - конвертация значений из перечисления в `one-hot encoding`
  - [CBitSetVectorizationLayer](DiscreteFeaturesLayers/BitSetVectorizationLayer.md) - конвертация `bitset`'ов в векторы из нулей и единиц
- Рекуррентные слои:
  - [CLstmLayer](LstmLayer.md) - реализация Long Short-Term Memory
  - [CGruLayer](GruLayer.md) - реализация Gated Recurrent Unit
- [Условное случайное поле (CRF)](CrfLayers/README.md):
  - [CCrfLayer](CrfLayers/CrfLayer.md) - условное случайное поле
  - [CCrfLossLayer](CrfLayers/CrfLossLayer.md) - функция потерь для обучения условного случайного поля
  - [CBestSequenceLayer](CrfLayers/BestSequenceLayer.md) - поиск наиболее вероятных последовательностей в результатах работы условного случайного поля
- [Connectionist Temporal Classification (CTC)](CtcLayers/README.md):
  - [CCtcLossLayer](CtcLayers/CtcLossLayer.md) - функция потерь
  - [CCtcDecodingLayer](CtcLayers/CtcDecodingLayer.md) - поиск наиболее вероятных последовательностей в ответах CTC
- Оценка качества классификации:
  - [CAccuracyLayer](QualityControlLayers/AccuracyLayer.md) - подсчет доли правильно классифицированных объектов
  - [CPrecisionRecallLayer](QualityControlLayers/PrecisionRecallLayer.md) - подсчет числа правильно классифицированных объектов обоих классов в бинарной классификации
  - [CConfusionMatrixLayer](QualityControlLayers/ConfusionMatrixLayer.md) - подсчет матрицы ошибок (`Confusion Matrix`) для многоклассовой классификации
