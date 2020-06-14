# Класс CTransposedConvLayer

<!-- TOC -->

- [Класс CTransposedConvLayer](#класс-ctransposedconvlayer)
    - [Настройки](#настройки)
        - [Размеры фильтров](#размеры-фильтров)
        - [Шаг свертки](#шаг-свертки)
        - [Дополнительные столбцы и колонки (padding)](#дополнительные-столбцы-и-колонки-padding)
        - [Разреженная свертка](#разреженная-свертка)
        - [Использование свободных членов](#использование-свободных-членов)
    - [Обучаемые параметры](#обучаемые-параметры)
        - [Фильтры](#фильтры)
        - [Свободные члены](#свободные-члены)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, выполняющий операцию, обратную свертке, над набором двумерных многоканальных изображений. Эта операция в различной литературе может называться `transposed convolution`, `deconvolution` или `up-convolution`. Поддерживает `padding` и `dilated convolution`.

## Настройки

### Размеры фильтров

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
void SetFilterCount( int filterCount );
```

Устанавливает количество и размеры фильтров.

### Шаг свертки

```c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
```

Устанавливает шаг свёртки. По умолчанию эти значения равны `1`.

### Дополнительные столбцы и колонки (padding)

```c++
void SetPaddingHeight( int paddingHeight );
void SetPaddingWidth( int paddingWidth );
```

Дополнительные строки и/или столбцы, которые будут срезаны у результата. Например, при `SetPaddingWidth( 1 );` у изображения будут удалены 2 колонки (по одной слева и справа). По умолчанию равны `0`.

### Разреженная свертка

```c++
void SetDilationHeight( int dilationHeight );
void SetDilationWidth( int dilationWidth );
```

Поддержка "разреженных" сверток. Это случаи, когда соседние пиксели фильтра разворачиваются не в соседние пиксели результата, а пиксели на расстоянии `dilation`. По умолчанию равны `1`, что соответствует неразреженной свертке.

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

Фильтры представляют собой [блоб](../DnnBlob.md) размера

- `BatchLength` равен `1`;
- `BatchWidth` равен `Channels * Depth` у входов;
- `ListSize` равен `1`;
- `Height` равен `GetFilterHeight()`;
- `Width` равен `GetFilterWidth()`;
- `Depth` равен `1`;
- `Channels` равен `GetFilterCount()`.

### Свободные члены

```c++
CPtr<CDnnBlob> GetFreeTermData() const;
```

Свободные члены представляют собой блоб, имеющий суммарный размер `GetFilterCount()`.

## Входы

На каждый вход подается блоб с набором изображений:

- `BatchLength * BatchWidth * ListSize` - количество изображений в наборе;
- `Height` - высота изображений;
- `Width` - ширина изображений;
- `Depth * Channels` - количество каналов у изображений.

Размеры блобов всех входов должны совпадать.

## Выходы

Для каждого входа соответствующий выход содержит блоб с результатом.

Блоб с результатами имеет следующие размеры:

- `BatchLength` равен `BatchLength` входа;
- `BatchWidth` равен `BatchWidth` входа;
- `ListSize` равен `ListSize` входа;
- `Height` рассчитывается относительно входа по формуле  
`StrideHeight * (Height - 1) + (FilterHeight - 1) * DilationHeight + 1 - 2 * PaddingHeight`;
- `Width` рассчитывается относительно входа по формуле  
`StrideWidth * (Width - 1) + (FilterWidth - 1) * DilationWidth + 1 - 2 * PaddingWidth`;
- `Depth` равен `1`;
- `Channels` равен `GetFilterCount()`.
