# Класс CTiedEmbeddingsLayer

<!-- TOC -->

- [Класс CTiedEmbeddingsLayer](#класс-ctiedembeddingslayer)
  - [Настройки](#настройки)
    - [EmbeddingsLayerName](#embeddingslayername)
    - [ChannelIndex](#channelindex)
  - [Входы](#входы)
  - [Выходы](#выходы)

<!-- /TOC -->

Класс реализует слой связанных эмбеддингов:

## Настройки

### EmbeddingsLayerName

```c++
void SetEmbeddingsLayerName( const char* name )
```
Использовать слой эмбеддингов `name`.


### ChannelIndex
```c++
void SetChannelIndex( int val );
```
Использовать `val` канал эмбеддингов.

## Входы

На каждый вход подаётся блоб, содержащий `BatchLength * BatchWidth * ListSize` объектов размера `1 * 1 * 1 * EmbeddingsSize`. Где `EmbeddingsSize` - размер вектора эмбеддингов.

## Выходы

Каждый выход содержит блоб, содержащий `BatchLength * BatchWidth * ListSize` объектов размера `1 * 1 * 1 * EmbeddingsCount`. Где `EmbeddingsCount` - количество эмбеддингов.
