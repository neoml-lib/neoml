# Класс CProjectionPoolingLayer

<!-- TOC -->

- [Класс CProjectionPoolingLayer](#класс-cprojectionpoolinglayer)
    - [Настройки](#настройки)
        - [Направление проецирования](#направление-проецирования)
        - [Сохранение размера блоба](#сохранение-размера-блоба)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой, выполняющий `Projection Pooling` над набором двумерных многоканальных изображений.

`Projection Pooling` вычисляет средние значения вдоль высоты или ширины блоба

## Настройки

### Направление проецирования

```c++
// Projection direction
enum TDirection {
    // Along BD_Width
    D_ByRows,
    // Along BD_Height
    D_ByColumns,

    D_EnumSize
};

// Projection direction
void SetDirection( TDirection _direction );
```

По умолчанию `D_ByRows`.

### Сохранение размера блоба

```c++
void SetRestoreOriginalImageSize( bool flag );
```

Если `true` то выходной блоб будет иметь ту же форму, что и у входа, и средние значения будут скопированы вдоль направления проецирования.
Если `false` то размер оси проецирования у выходного блоба будет равен `1`.
По умолчанию `false`.

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

На единственный вход подается блоб с набором изображений:

- `BatchLength * BatchWidth * ListSize` - количество изображений в наборе;
- `Height` - высота изображений;
- `Width` - ширина изображений;
- `Depth * Channels` - количество каналов у изображений.

## Выходы

Единственный выход содержит блоб размера:

- `BatchLength` равный `BatchLength` входа;
- `BatchWidth` равный `BatchWidth` входа;
- `ListSize` равный `ListSize` входа;
- `Height` равен `1` если флаг `RestoreOriginalImageSize` не установлен и направление проецирования равно `D_ByColumns`, иначе равен `Height` входа;
- `Width` равен `1` если флаг `RestoreOriginalImageSize` не установлен и направление проецирования равно `D_ByRows`, иначе равен `Width` входа;
- `Depth` равен `Depth` входа;
- `Channels` равен `Channels` входа.
