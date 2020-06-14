# Класс CBaseLayer

<!-- TOC -->

- [Класс CBaseLayer](#класс-cbaselayer)
    - [Конструктор](#конструктор)
    - [Имя слоя](#имя-слоя)
    - [Сеть](#сеть)
    - [Присоединение к другим слоям](#присоединение-к-другим-слоям)
    - [Количество входов](#количество-входов)
    - [Информация о входах](#информация-о-входах)
    - [Управление обучением](#управление-обучением)
    - [Множитель скорости сходимости](#множитель-скорости-сходимости)
    - [Множитель регуляризатора](#множитель-регуляризатора)

<!-- /TOC -->

Базовый класс слоя. Содержит логику установки соединений с другими слоями и взаимодействия с сетью ([`CDnn`](#Dnn.md)). Классы для [всех остальных слоёв](README.md/#список-слоёв) наследуются от этого класса.

## Конструктор

```c++
CBaseLayer( IMathEngine& mathEngine, const char* name, bool isLearnable );
```

Этот метод используется внутри конструкторов для специализированных слоёв.

Создаёт новый слой. При создании необходимо передать ссылку на используемый вычислительный движок (один и тот же для всех слоёв одной сети). Можно также задать имя слоя и указать, имеет ли слой обучаемые веса.

## Имя слоя

```c++
const char* GetName() const;
void SetName( const char* name );
```

Меняет имя слоя. Изменять имя разрешено только для слоёв, не входящих в сеть.

## Сеть

```c++
const CDnn* GetDnn() const;
CDnn* GetDnn();
```

Получить указатель на сеть, в которой находится слой. Возвращает `0`, если слой не находится в какой-либо сети.

## Присоединение к другим слоям

```c++
void Connect( int inputNumber, const char* input, int outputNumber = 0 );
void Connect( int inputNumber, const CBaseLayer& layer, int outputNumber = 0 );
void Connect( const char* input );
void Connect( const CBaseLayer& layer );
```

Присоединить вход номер `inputNumber` к выходу номер `outputNumber` слоя `layer` (или слоя с именем `input`).

## Количество входов

```c++
int GetInputCount() const;
```

## Информация о входах

```c++
const char* GetInputName(int number) const;

int GetInputOutputNumber(int number) const;
```

Получить описания входов.

## Управление обучением

```c++
void DisableLearning();
void EnableLearning();
bool IsLearningEnabled() const;
```

Если обучение отключено, то слой не будет обучаться при вызовах `CDnn::RunAndLearnOnce`.

## Множитель скорости сходимости

```c++
float GetBaseLearningRate() const;
void SetBaseLearningRate( float rate );
```

Базовая скорость обучения слоя. На нее домножается `learningRate` из оптимизатора сети. Используется для изменения скорости обучения одного слоя относительно остальных.

## Множитель регуляризатора

```c++
float GetBaseL1RegularizationMult() const;
void SetBaseL1RegularizationMult(float mult);
float GetBaseL2RegularizationMult() const;
void SetBaseL2RegularizationMult( float mult );
```

Базовые коэффициенты регуляризации. На них домножаются соответствующие коэффициенты из оптимизатора сети.
