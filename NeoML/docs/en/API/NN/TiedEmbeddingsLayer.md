# CTiedEmbeddingsLayerd Class

<!-- TOC -->

- [CTiedEmbeddingsLayerd Class](#ctiedembeddingslayerd-class)
  - [Settings](#settings)
    - [EmbeddingsLayerName](#embeddingslayername)
    - [ChannelIndex](#channelindex)
  - [Inputs](#inputs)
  - [Outputs](#outputs)

<!-- /TOC -->

Class implements tied embeddings layer.

## Settings

### EmbeddingsLayerName

```c++
void SetEmbeddingsLayerName( const char* name )
```
Embeddings layer `name`.


### ChannelIndex
```c++
void SetChannelIndex( int val );
```
Embedding channel index `val`

## Inputs

Each input accepts a blob with `BatchLength * BatchWidth * ListSize` objects with size `1 * 1 * 1 * EmbeddingsSize`.

`EmbeddingsSize` - size of the embeddings vector.

## Outputs

Each output contains a blob with `BatchLength * BatchWidth * ListSize` objects with size  `1 * 1 * 1 * EmbeddingsCount`.

`EmbeddingsCount` - embeddings count.
