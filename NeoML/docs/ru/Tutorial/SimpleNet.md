# Пример простой сети

<!-- TOC -->

- [Пример простой сети](#пример-простой-сети)
	- [Создание объекта нейронной сети](#создание-объекта-нейронной-сети)
	- [Создание слоёв](#создание-слоёв)
	- [Создание блобов для данных](#создание-блобов-для-данных)
	- [Обучение сети](#обучение-сети)
	- [Оценка результата](#оценка-результата)
		- [Вывод](#вывод)

<!-- /TOC -->

В этом примере мы используем библиотеку **NeoML** для создания и обучения простой нейросети, решающей задачу классификации на классическом датасете [MNIST](https://ru.wikipedia.org/wiki/MNIST_(%D0%B1%D0%B0%D0%B7%D0%B0_%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85)).

## Создание объекта нейронной сети

Нейронная сеть реализуется с помощью класса [CDnn](../API/NN/Dnn.md). Перед созданием нового объекта нейронной сети нужно также создать вычислительный движок (интерфейс [IMathEngine](../API/NN/MathEngine.md)) и генератор случайных чисел, который будет использоваться для всех случайных величин во время инициализации слоёв и работы сети.

В этом примере мы используем вычислительный движок по умолчанию, который работает на CPU. Есть возможность также вычислять на GPU или задавать дополнительные настройки (см. описание [IMathEngine](../API/NN/MathEngine.md)).

```c++
// Вычислительный движок по умолчанию, производящий вычисления на CPU.
IMathEngine& mathEngine = GetDefaultCpuMathEngine();
// Генератор случайных чисел.
CRandom random( 451 );
// Нейронная сеть.
CDnn net( random, mathEngine );
```

## Создание слоёв

Теперь мы можем создать, настроить и подключить к сети необходимый набор слоёв. Обратите внимание, что имена слоёв в одной сети не могут совпадать.

Для решения задачи классификации нам понадобится:

1. Два [входных слоя](../API/NN/IOLayers/SourceLayer.md) — первый с изображениями, второй с информацией о правильных классах.
2. Для собственно классификации изображений дважды используем комбинацию полносвязного слоя с функцией активации `ReLU`.
3. Вычислим функцию потери на выходе сети и втором входе (с правильными метками). Целью обучения будет минимизация функции потерь.

В данном примере мы лишь проверяем возможность обучения сети; сами результаты классификации не возвращаются.
Чтобы извлечь результаты, нужно к выходу сети (в данном случае слоя `relu2`) подключить выходной слой [CSinkLayer](../API/NN/IOLayers/SinkLayer.md).

```c++
// Входной слой для данных.
CPtr<CSourceLayer> data = new CSourceLayer( mathEngine );
data->SetName( "data" ); // Задание имени слоя, уникального в рамках сети.
net.AddLayer( *data ); // Добавление слоя в сеть.

// Входной слой для правильных ответов.
CPtr<CSourceLayer> label = new CSourceLayer( mathEngine );
label->SetName( "label" );
net.AddLayer( *label );

// Первый полносвязный слой размера 1024.
CPtr<CFullyConnectedLayer> fc1 = new CFullyConnectedLayer( mathEngine );
fc1->SetName( "fc1" );
fc1->SetNumberOfElements( 1024 ); // Задание количества элементов в полносвязном слое.
fc1->Connect( *data ); // Присоединение к предыдущему слою.
net.AddLayer( *fc1 );

// Функция активации.
CPtr<CReLULayer> relu1 = new CReLULayer( mathEngine );
relu1->SetName( "relu1" );
relu1->Connect( *fc1 );
net.AddLayer( *relu1 );

// Второй полносвязный слой размера 512.
CPtr<CFullyConnectedLayer> fc2 = new CFullyConnectedLayer( mathEngine );
fc2->SetName( "fc2" );
fc2->SetNumberOfElements( 512 );
fc2->Connect( *relu1 );
net.AddLayer( *fc2 );

// Функция активации.
CPtr<CReLULayer> relu2 = new CReLULayer( mathEngine );
relu2->SetName( "relu2" );
relu2->Connect( *fc2 );
net.AddLayer( *relu2 );

// Третий полносвязный слой размера, равного числу классов (10).
CPtr<CFullyConnectedLayer> fc3 = new CFullyConnectedLayer( mathEngine );
fc3->SetName( "fc3" );
fc3->SetNumberOfElements( 10 );
fc3->Connect( *relu2 );
net.AddLayer( *fc3 );

// Кросс-энтропия. Внутри этого слоя уже вычисляется SoftMax,
// потому добавлять слой Softmax перед ним не нужно.
CPtr<CCrossEntropyLossLayer> loss = new CCrossEntropyLossLayer( mathEngine );
loss->SetName( "loss" );
loss->Connect( 0, *fc3 ); // Вход #1: ответ сети.
loss->Connect( 1, *label ); // Вход №2: правильный ответ.
net.AddLayer( *loss );
```

## Создание блобов для данных

Все данные разобьём на батчи по 100 изображений, и таким образом из выборки в 60000 изображений получаем 600 итераций на одну эпоху обучения нейросети.

Используем методы класса [CDnnBlob](../API/NN/DnnBlob.md) для создания и заполнения блобов с входными изображениями и их разметкой по классам.

```c++
const int batchSize = 100; // Размер батча.
const int iterationPerEpoch = 600; // Данные для обучения содержат 60000 картинок (600 батчей).

// Блоб для данных, содержащий 100 картинок из MNIST (1 канал, высоты 29 и ширины 28).
CPtr<CDnnBlob> dataBlob = CDnnBlob::Create2DImageBlob( mathEngine, CT_Float, 1, batchSize, 29, 28, 1 );
// Блоб для меток, содержащий 100 векторов длины 10, с единичкой на нужном ответе (one-hot).
CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, batchSize, 10 );

// Передача блобов во входные слои.
data->SetBlob( dataBlob );
label->SetBlob( labelBlob );
```

## Обучение сети

Для обучения сети вызовем метод `RunAndLearnOnce()` объекта сети. На каждой итерации мы получим значение функции потерь с помощью функции `GetLastLoss()` и сложим их,
чтобы найти общую ошибку за эпоху обучения.

Входные данные пересортируются случайным образом после каждой эпохи.

```c++
for( int epoch = 1; epoch < 15; ++epoch ) {
    float epochLoss = 0; // Суммарная ошибка за эпоху.
    for( int iter = 0; iter < iterationPerEpoch; ++iter ) {
        // В trainData скрыта передача данных из датасета в блоб.
        trainData.GetSamples( iter * batchSize, dataBlob );
        trainData.GetLabels( iter * batchSize, labelBlob );

        net.RunAndLearnOnce(); // Запуск итерации обучения.
        epochLoss += loss->GetLastLoss(); // Прибавляем значение ошибки с последнего шага.
    }

    ::printf( "Epoch #%02d    avg loss: %f\n", epoch, epochLoss / iterationPerEpoch );
    trainData.ReShuffle( random ); // Перемешаем порядок данных.
}
```

## Оценка результата

Для проверки результата обучения используем тестовую выборку из 10000  других изображений, также разделённых на батчи по 100.
Для запуска нейросети без обучения используем функцию `RunOnce()` и подсчитаем общую ошибку на всех батчах.

```c++
float testDataLoss = 0;
// Данные для проверки содержат 10000 картинок (100 батчей).
for( int testIter = 0; testIter < 100; ++testIter ) {
    testData.GetSamples( testIter * batchSize, dataBlob );
    testData.GetLabels( testIter * batchSize, labelBlob );
    net.RunOnce();
    testDataLoss += loss->GetLastLoss();
}

::printf( "\nTest data loss: %f\n", testDataLoss / 100 );
```

### Вывод

На тестовом запуске описанной выше сети мы получили такой вывод:

```
Epoch #01    avg loss: 0.519273
Epoch #02    avg loss: 0.278983
Epoch #03    avg loss: 0.233433
Epoch #04    avg loss: 0.204021
Epoch #05    avg loss: 0.182192
Epoch #06    avg loss: 0.163927
Epoch #07    avg loss: 0.149121
Epoch #08    avg loss: 0.136408
Epoch #09    avg loss: 0.126139
Epoch #10    avg loss: 0.116643
Epoch #11    avg loss: 0.108768
Epoch #12    avg loss: 0.101515
Epoch #13    avg loss: 0.095355
Epoch #14    avg loss: 0.089328

Test data loss: 0.100225
```

Как видим, ошибка на обучающей выборке постепенно уменьшается в процессе обучения. Ошибка на тестовой выборке также в приемлемых границах.
