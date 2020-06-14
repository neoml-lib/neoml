# Класс CRleConvLayer

<!-- TOC -->

- [Класс CRleConvLayer](#класс-crleconvlayer)
    - [Формат RLE](#формат-rle)
        - [Пример](#пример)
    - [Настройки](#настройки)
        - [Размеры фильтров](#размеры-фильтров)
        - [Шаг свертки](#шаг-свертки)
        - [Значения пикселей в RLE](#значения-пикселей-в-rle)
        - [Использование свободных членов](#использование-свободных-членов)
    - [Обучаемые параметры](#обучаемые-параметры)
        - [Фильтры](#фильтры)
        - [Свободные члены](#свободные-члены)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, выполняющий операцию свертки над набором бинарных одноканальных изображений в формате [RLE](#формат-rle).

## Формат RLE

В формате RLE изображение представляется значением фона (`GetNonStrokeValue()`) и набором горизонтальных штрихов, пиксели в которых заполнены значениями `GetStrokeValue()`.

```c++
static const int MaxRleConvImageWidth = 64;

// штрих в RLE представлении изображения
struct CRleStroke {
	short Start;	// начало штриха
	short End;		// конец штриха (первая позиция ЗА штрихом)

    // Специальный штрих, означающий конец строки
	static CRleStroke Sentinel() { return { SHRT_MAX, -1 }; }
};

static const CRleStroke Sentinel = { SHRT_MAX, -1 };

struct CRleImage {
	int StrokesCount; // Количество штрихов в изображении
	int Height; // Высота изображения. Может быть меньше высоты блоба.
                // В таком случае верхние (blob.GetHeight() - Height) / 2
                // и нижние (blob.GetHeight() - Height + 1) / 2 строк будут заполнены GetNonStrokeValue()
	int Width; // Ширина изображения. Может быть меньше ширины блоба.
                // В таком случае левые (blob.GetWidth() - Width) / 2
                // и правые (blob.GetWidth() - Width + 1) / 2 столбцов будут заполнены GetNonStrokeValue()
	CRleStroke Stub; // RESERVED
	CRleStroke Lines[1]; // Массив линий. Т.к. координаты штрихов хранятся в short, гарантируется, что размер любого RLE-изображения не превысит размера float-буфера, необходимого для его хранения
};
```

### Пример

```c++
// Закодируем в RLE следующее изображение:
// 01110
// 00000
// 01010
// 00110.

CPtr<CDnnBlob> imageBlob = CDnnBlob::Create2DImageBlob(GetDefaultCpuMathEngine(), CT_Float, 1, 1, 4, 5, 1);
CArray<float> imageBuff;
imageBuff.Add(0, 4 * 5);
CRleImage* rleImage = reinterpret_cast<CRleImage*>(imageBuff.GetPtr());
rleImage->Height = 4;
rleImage->Width = 3; // левая и правая колонки заполнены нулями.
rleImage->StrokesCount = 8; // 4 штриха на картинке + 4 обозначения окончания строки.
rleImage->Lines[0] = CRleStroke{ 0, 3 }; // 3 единицы в первой строке. Координаты штриха записываются относительно rleImage->Width, а не ширины блоба!
rleImage->Lines[1] = CRleStroke::Sentinel(); // Конец первой строки.
rleImage->Lines[2] = CRleStroke::Sentinel(); // Вторая строка пустая.
rleImage->Lines[3] = CRleStroke{ 0, 1 }; // Первая единица на третьей строке.
rleImage->Lines[4] = CRleStroke{ 2, 3 }; // Вторая единица на третьей строке.
rleImage->Lines[5] = CRleStroke::Sentinel(); // Конец третьей строки.
rleImage->Lines[6] = CRleStroke{ 1, 3 };
rleImage->Lines[7] = CRleStroke::Sentinel();
imageBlob->CopyFrom(imageBuff.GetPtr());

// Для записи нескольких изображений n-ю картинку надо расположить в массиве imageBuff
// по сдвигу, равному размеру изображения * (n-1)
// CRleImage* nthRleImage = reinterpret_cast<CRleImage*>(imageBuff.GetPtr() + (4 * 5) * (n - 1));
```

## Настройки

### Размеры фильтров

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
void SetFilterCount( int filterCount );
```

### Шаг свертки

```c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
```

По умолчанию равны `1`.

### Значения пикселей в RLE

```c++
void SetStrokeValue( float _strokeValue );
void SetNonStrokeValue( float _nonStrokeValue );
```

Значения в штрихах RLE и вне штрихов. Подробнее о формате см. [выше](#формат-rle).

### Использование свободных членов

```c++
void SetZeroFreeTerm(bool isZeroFreeTerm);
```

Указывает, нужно ли использовать вектор свободных членов. Если передать `true`, содержимое вектора будет заполнено нулями, и он не будет обучаться. По умолчанию `false`.

## Обучаемые параметры

### Фильтры

```c++
CPtr<CDnnBlob> GetFilterData() const;
```

Фильтры представляют собой [блоб](../DnnBlob.md) размера:

- `BatchLength * BatchWidth * ListSize` равен `GetFilterCount()`;
- `Height` равен `GetFilterHeight()`;
- `Width` равен `GetFilterWidth()`;
- `Depth` и `Channels` равны `1`.

### Свободные члены

```c++
CPtr<CDnnBlob> GetFreeTermData() const;
```

Свободные члены представляют собой блоб суммарного размера `GetFilterCount()`.

## Входы

На каждый вход подается блоб с набором изображений:

- `BatchLength * BatchWidth * ListSize` - количество изображений в наборе;
- `Height` - высота изображений;
- `Width` - ширина изображений, не должна превышать `64`;
- `Depth` - глубина, должна быть равна `1`;
- `Channels` - каналы, должны быть равны `1`.

Размеры блобов всех входов должны совпадать.

Изображения в блобах записаны в формате RLE.


## Выходы

Для каждого входа соответствующий выход содержит блоб с результатом свертки. Результаты хранятся в том же формате, что и у обычной свертки ([`CConvLayer`](ConvLayer.md)).

Блоб с результатами имеет следующие размеры:

- `BatchLength` равный `BatchLength` входа;
- `BatchWidth` равный `BatchWidth` входа;
- `ListSize` равный `ListSize` входа;
- `Height` рассчитывается относительно входа по формуле  
`(Height - FilterHeight)/StrideHeight + 1`;
- `Width` рассчитывается относительно входа по формуле  
`(Width - FilterWidth)/StrideWidth + 1`;
- `Depth` равен `1`;
- `Channels` равен `GetFilterCount()`.
