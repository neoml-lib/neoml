# Класс CDnnBlob

<!-- TOC -->

- [Класс CDnnBlob](#класс-cdnnblob)
    - [Блоб как тензор](#блоб-как-тензор)
        - [Поддерживаемые типы данных](#поддерживаемые-типы-данных)
        - [Представление в памяти](#представление-в-памяти)
    - [Создание блобов](#создание-блобов)
        - [Блобы для данных](#блобы-для-данных)
        - [Блобы-окна](#блобы-окна)
        - [Блобы для математических операций](#блобы-для-математических-операций)
    - [Получение размеров блоба](#получение-размеров-блоба)
    - [Обмен данными](#обмен-данными)
    - [Создание копий блобов](#создание-копий-блобов)
    - [Изменение данных в блобе](#изменение-данных-в-блобе)
    - [Соединение блобов](#соединение-блобов)
    - [Разбиение блоба](#разбиение-блоба)
	    - [Параметры](#параметры)

<!-- /TOC -->

Класс, используемый для хранения и передачи данных в нейронных сетях.

## Блоб как тензор

Блоб представляет собой 7-мерный массив, каждая размерность которого имеет определенное значение:

- `BatchLength` - "временная" шкала, используемая для обозначения последовательностей данных; обычно применяется в рекуррентных сетях;
- `BatchWidth` - батч, используется для одновременной передачи нескольких не связанных между собой объектов;
- `ListSize` - размерность, используемая для обозначения того, что объекты связаны между собой (например, это могут быть пиксели, извлеченные из одного изображения), но при этом не являются последовательностью;
- `Height` - высота, используется при работе с матрицами или изображениями;
- `Width` - ширина, используется при работе с матрицами или изображениями;
- `Depth` - глубина, используется при работе с трехмерными изображениями;
- `Channels` - каналы, используется при работе с многоканальными изображениями, а также при работе с одномерными векторами.

### Поддерживаемые типы данных

Поддерживаются два типа данных: с плавающей точкой (`CT_Float`) и целочисленный (`CT_Int`). В обоих случаях используются 32-битные типы данных. Если где-либо в этой документации описание блоба не содержит явного указания типа данных, то подразумеваются данные с плавающей точкой.

### Представление в памяти

В памяти данные лежат таким образом, что соседние элементы являются соседними по оси **Channels**. Для перехода на следующую координату по оси **Depth** нужно сдвинуться на число элементов, равное числу каналов и т.д. в порядке, перечисленном выше (так называемый **channel-last ordering**).

## Создание блобов

Для создания блобов предоставляется набор статических функций.

### Блобы для данных

```c++
static CDnnBlob* CreateDataBlob( IMathEngine& mathEngine, TDnnType type, int batchLength, int batchWidth, int channelsCount );
```

Создание блоба с данными типа `type`, представляющего из себя `batchWidth` последовательностей, длины `batchLength`, каждый элемент которых имеет `channelsCount` каналов.

```c++
static CDnnBlob* CreateListBlob( IMathEngine& mathEngine, TDnnType type, int batchLength, int batchWidth, int listSize, int channelsCount );
```

Функция аналогична `CreateDataBlob`, но каждая последовательность состоит из списков длиной `listSize`.

```c++
static CDnnBlob* Create2DImageBlob( IMathEngine& mathEngine, TDnnType type, int batchLength, int batchWidth, int imageHeight, int imageWidth, int channelsCount );
```

Создание блоба с данными типа `type`, представляющего из себя `batchWidth` последовательностей, длины `batchLength`, каждый элемент которых является двумерным изображением высоты `imageHeight` и ширины `imageWidth`, имеющим `channelsCount` каналов.

```c++
static CDnnBlob* Create3DImageBlob( IMathEngine& mathEngine, TDnnType type, int batchLength, int batchWidth, int imageHeight, int imageWidth, int imageDepth, int channelsCount );
```

Функция аналогична `Create2DImageBlob`, но с трехмерными изображениями глубины `imageDepth`.

### Блобы-окна

```c++
static CDnnBlob* CreateWindowBlob( const CPtr<CDnnBlob>& parent, int windowSize = 1 );
```

Создание блоба, который является некоторым окном длины `windowSize` над блобом `parent`. Данный блоб имеет `BatchLength`, равный `windowSize`, а остальные размерности и тип данных равны соответствующим у блоба `parent`. Данный блоб не имеет самостоятельного буфера, а является лишь дополнительным указателем на память родительского, и становится **невалидным**, если родительский блоб будет разрушен. Пользователь должен следить за тем, чтобы блобы-окна не использовались после разрушения родительского блоба.

```c++
CDnnBlob* GetParent();
const CDnnBlob* GetParent() const;
```

Получение указателя на родительский блоб. Возвращает `0`, если блоб не является окном.

```c++
CDnnBlob* GetOwner();
const CDnnBlob* GetOwner() const;
```

Получение указателя на блоб, который владеет данными. Возвращает `this`, если текущий блоб не был создан при помощи `CreateWindow`.

```c++
int GetParentPos() const;
void SetParentPos( int pos );
void ShiftParentPos( int shift );
```

Смещение окна в рамках родительского блоба. Все позиции и сдвиги интерпретируются как координаты по оси `BatchLength`.

### Блобы для математических операций

```c++
static CDnnBlob* CreateTensor(IMathEngine& mathEngine, TDnnType type, std::initializer_list<int> dimensions);

// CreateVector(x) аналогичен вызову CreateTensor({x})
static CDnnBlob* CreateVector(IMathEngine& mathEngine, TDnnType type, int vectorSize);

// CreateMatrix(x, y) аналогичен вызову CreateTensor({x, y})
static CDnnBlob* CreateMatrix(IMathEngine& mathEngine, TDnnType type, int matrixHeight, int matrixWidth);
```

Создание n-мерных, одномерных и двумерных блобов. В случае `CreateTensor` длина списка `dimensions` должна быть не более 7.

## Получение размеров блоба

```c++
int GetBatchLength() const;
int GetBatchWidth() const;
int GetListSize() const;
int GetHeight() const;
int GetWidth() const;
int GetDepth() const;
int GetChannelsCount() const;

int DimSize( int d ) const;
int DimSize( TBlobDim d ) const;
```

Получение размера блоба вдоль одной из осей.

```c++
int GetDataSize() const;
```

Получить полный размер блоба (произведение размеров по всем 7 осям).

```c++
int GetObjectCount() const;
```

Получить количество объектов в блобе. Возвращает произведение `BatchLength * BatchWidth * ListSize`. Используется вместо `BatchWidth` в тех местах, где не требуется отдельная обработка `BatchLength` и `ListSize`.

```c++
int GetObjectSize() const;
```

Получить размер одного объекта в блобе. Возвращает произведение `Height * Width * Depth * Channels`. Используется в тех местах, где любые объекты интерпретируются как одномерные векторы.

```c++
int GetGeometricalSize() const;
```

Получение "геометрии" блоба. Возвращает произведение `Height * Width * Depth`. Используется в тех местах, где нет смысла разделять размерности `Height`, `Width` и `Depth`.

```c++
bool HasEqualDimensions( const CDnnBlob* other ) const;
```

Проверка того, что другой блоб имеет те же размеры.

## Обмен данными

```c++
template<class T = float>
void CopyFrom( const T* src );

template<class T = float>
void CopyTo( T* dst ) const;
template<class T = float>
void CopyTo( T* dst, int size ) const;
```

Обмен типизированными данными с внешним кодом. Если размер не указан явно, копируется всё содержимое блоба (`GetDataSize`).

```c++
void CopyFrom( const CDnnBlob* other );
```

Скопировать данные из другого блоба. Требуется совпадение размеров и типа данных.

```c++
void TransposeFrom( const CDnnBlob* other, int d1, int d2 );
```

Скопировать данные из другого блоба, поменяв две размерности местами.

## Создание копий блобов

```c++
CDnnBlob* GetCopy() const;
```

Создает независимую от `this` копию блоба.

```c++
CDnnBlob* GetClone() const;
CDnnBlob* GetClone( TDnnType type ) const;
```

Создает блоб того же размера, но с неинициализированными данными. Во втором случае также изменяет тип данных.

```c++
CDnnBlob* GetTransposed( int d1, int d2 ) const;
```

Получить копию блоба, где 2 размерности поменяли местами. Транспонирует данные в памяти.

## Изменение данных в блобе

```c++
void Add( const CDnnBlob* other );
```

Поэлементное прибавление значений из другого блоба. Необходимо совпадение размера блобов.

```c++
void Clear();
void ClearObject( int num );
```

Занулить элементы блоба или конкретного объекта.

```c++
template<class T = float>
void Fill( typename CDnnType<T>::TDataType value );
template<class T = float>
void FillObject( int num, typename CDnnType<T>::TDataType value );
```

Заполнить элементы блоба или конкретного объекта данным значением.

```c++
template<class T>
T* GetBuffer( int pos, int size, bool exchange );
void ReleaseBuffer( void* ptr, bool exchange );
```

Методы для работы непосредственно с памятью (для блобов на cpu) или с копией данных (для блобов в виде-памяти).
Параметр `exchange` позволяет избежать лишних обменов данными (например, не читать из видео-памяти данные неинициализированного блоба или не отправлять обратно то, что не менялось).
Должны вызываться строго в LIFO-порядке (соблюдать вложенность вызовов).
Для упрощения работы с данными методами можно использовать вспомогательный класс `CDnnBlobBuffer`.

## Соединение блобов

```c++
static void MergeByChannels( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByDepth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByWidth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByHeight( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByListSize( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByBatchWidth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByBatchLength( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByObject( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );

static void MergeByDim( IMathEngine& mathEngine, TBlobDim d, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
static void MergeByDim( IMathEngine& mathEngine, int d, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
```

Соединить блобы вдоль одной из размерностей.

## Разбиение блоба

```c++
static void SplitByChannels( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByDepth( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByWidth( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByHeight( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByListSize( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByBatchWidth( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByBatchLength( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByObject( IMathEngine& mathEngine, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );

static void SplitByDim( IMathEngine& mathEngine, TBlobDim d, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
static void SplitByDim( IMathEngine& mathEngine, int d, const CPtr<CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
```

Разбивает блоб на меньшие части вдоль одной из размерностей. Метод `SplitByObject` разбивает вдоль произведения размерностей `BatchLength * BatchWidth * ListSize`.

### Параметры

* *mathEngine* - ссылка на вычислительный движок;
* *from* - исходный блоб;
* *to* - массив блобов-частей, в которые нужно записать результаты разбиения. Размеры блобов в этом массиве определяют разбиение; все размерности, кроме той, по которой происходит разбиение, должны совпадать. Суммарная размерность, по которой происходит разбиение, должна быть равна этой размерности у исходного блоба.
