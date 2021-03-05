# Класс CTiedEmbeddingsLayer

<!-- TOC -->

- [Класс CTiedEmbeddingsLayer](#класс-ctiedembeddingslayer)
  - [Настройки](#настройки)
    - [EmbeddingsLayerName](#embeddingslayername)
    - [ChannelIndex](#channelindex)
  - [Обучаемые параметры](#обучаемые-параметры)
  - [Входы](#входы)
  - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует [слой связанных эмбеддингов](https://arxiv.org/pdf/1608.05859.pdf):

## Настройки

### EmbeddingsLayerName

```c++
void SetEmbeddingsLayerName( const char* name )
```
Использовать слой эмбеддингов `name`. Только слой CMultichannelLookupLayer разрешен.

### ChannelIndex

```c++
void SetChannelIndex( int val );
```
Использовать `val` канал эмбеддингов.

## Обучаемые параметры

Слой использует обучаемые параметры из слоя эмбеддингов.

## Входы

На каждый вход подается блоб с размерами:
- `BatchLength * BatchWidth * ListSize` равно числу объектов
- `Height`, `Width`, `Depth` равно `1`
- `Channels` равно `EmbeddingSize`

## Выходы

Количество выходов равно количеству входов.
Каждый выход содержит блоб с размерами:
- `BatchLength * BatchWidth * ListSize` объектов 
- `Height`, `Width`, `Depth` равно `1`
- `Channels` равно `EmbeddingsCount` - количество эмбеддингов.
