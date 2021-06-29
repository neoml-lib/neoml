# CTiedEmbeddingsLayer Class

<!-- TOC -->

- [CTiedEmbeddingsLayer Class](#ctiedembeddingslayer-class)
  - [Settings](#settings)
    - [EmbeddingsLayerName](#embeddingslayername)
    - [ChannelIndex](#channelindex)
  - [Trainable parameters](#trainable-parameters)
  - [Inputs](#inputs)
  - [Outputs](#outputs)

<!-- /TOC -->

Class implements [tied embeddings layer](https://arxiv.org/pdf/1608.05859.pdf).

## Settings

### EmbeddingsLayerName

```c++
void SetEmbeddingsLayerName( const char* name )
```
Embeddings layer `name`. Only [CMultichannelLookupLayer](DiscreteFeaturesLayers/MultichannelLookupLayer.md) is allowed.

### ChannelIndex

```c++
void SetChannelIndex( int val );
```
Embedding channel index.

## Trainable parameters

This layer uses shared parameters from the embeddings layer.

## Inputs

Each input accepts a blob of the dimensions:
- `BatchLength * BatchWidth * ListSize` is equal to number of objects
- `Height`, `Width`, `Depth` are equal to `1`
- `Channels` is equal to `EmbeddingSize`

## Outputs

The number of outputs is equal to the number of inputs.
Each output contains a blob with dimensions:
- `BatchLength * BatchWidth * ListSize` objects
- `Height`, `Width`, `Depth` are equal to `1`
- `Channels` is equal to `EmbeddingsCount`
