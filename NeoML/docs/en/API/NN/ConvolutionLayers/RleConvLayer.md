# CRleConvLayer Class

<!-- TOC -->

- [CRleConvLayer Class](#crleconvlayer-class)
    - [RLE format](#rle-format)
        - [Sample](#sample)
    - [Settings](#settings)
        - [Filters size](#filters-size)
        - [Convolution stride](#convolution-stride)
        - [Pixel values in RLE format](#pixel-values-in-rle-format)
        - [Using the free terms](#using-the-free-terms)
    - [Trainable parameters](#trainable-parameters)
        - [Filters](#filters)
        - [Free terms](#free-terms)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that performs convolution on a set of binary one-channel images in [RLE](#rle-format) format.

## RLE format

An RLE image is represented by the background value (`GetNonStrokeValue()`) and a set of horizontal strokes filled with `GetStrokeValue()`.

```c++
static const int MaxRleConvImageWidth = 64;

// A stroke in RLE format
struct CRleStroke {
	short Start;	// starting coordinate
	short End;		// ending coordinate (the first pixel that does NOT belong to the stroke)

    // A special stroke value for the end of the row
	static CRleStroke Sentinel() { return { SHRT_MAX, -1 }; }
};

static const CRleStroke Sentinel = { SHRT_MAX, -1 };

struct CRleImage {
	int StrokesCount; // the number of strokes in the image
	int Height; // the image height; may be smaller than the blob height,
                // in which case the top (blob.GetHeight() - Height) / 2
                // and the bottom (blob.GetHeight() - Height + 1) / 2 rows will be filled with GetNonStrokeValue()
	int Width; // the image width; may be smaller than the blob width,
               // in which case the left (blob.GetWidth() - Width) / 2
               // and the right (blob.GetWidth() - Width + 1) / 2 columns will be filled with GetNonStrokeValue()
	CRleStroke Stub; // RESERVED
	CRleStroke Lines[1]; // the lines array. As the strokes' coordinates are stored in short variables, it is guaranteed that the size of any RLE image will not exceed the size of a float buffer necessary to store it
};
```

### Sample

```c++
// We will encode the following image in RLE format:
// 01110
// 00000
// 01010
// 00110

CPtr<CDnnBlob> imageBlob = CDnnBlob::Create2DImageBlob(GetDefaultCpuMathEngine(), CT_Float, 1, 1, 4, 5, 1);
CArray<float> imageBuff;
imageBuff.Add(0, 4 * 5);
CRleImage* rleImage = reinterpret_cast<CRleImage*>(imageBuff.GetPtr());
rleImage->Height = 4;
rleImage->Width = 3; // the left and the right columns are filled with zeros
rleImage->StrokesCount = 8; // there are 4 strokes on the image + 4 end-of-rows are needed
rleImage->Lines[0] = CRleStroke{ 0, 3 }; // the 3-length stroke in the first row; the stroke coordinates are relative to rleImage->Width and NOT to the blob width
rleImage->Lines[1] = CRleStroke::Sentinel(); // the first row ends
rleImage->Lines[2] = CRleStroke::Sentinel(); // the second row is empty
rleImage->Lines[3] = CRleStroke{ 0, 1 }; // the first one pixel in the third row
rleImage->Lines[4] = CRleStroke{ 2, 3 }; // the second one pixel in the third row
rleImage->Lines[5] = CRleStroke::Sentinel(); // the third row ends
rleImage->Lines[6] = CRleStroke{ 1, 3 };
rleImage->Lines[7] = CRleStroke::Sentinel();
imageBlob->CopyFrom(imageBuff.GetPtr());

// If you want to store several images, place the nth image in the imageBuff array
// shifted by the image size * (n-1)
// CRleImage* nthRleImage = reinterpret_cast<CRleImage*>(imageBuff.GetPtr() + (4 * 5) * (n - 1));
```

## Settings

### Filters size

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
void SetFilterCount( int filterCount );
```

### Convolution stride

```c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
```

Sets the convolution stride. By default, the stride is `1`.

### Pixel values in RLE format

```c++
void SetStrokeValue( float _strokeValue );
void SetNonStrokeValue( float _nonStrokeValue );
```

The values to be used for the RLE strokes and outside them. See [above](#rle-format) for details on format.

### Using the free terms

```c++
void SetZeroFreeTerm(bool isZeroFreeTerm);
```

Specifies if the free terms should be used. If you set this value to `true`, the free terms vector will be set to all zeros and won't be trained. By default, this value is set to `false`.

## Trainable parameters

### Filters

```c++
CPtr<CDnnBlob> GetFilterData() const;
```

The filters are represented by a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to  the number of filters used (`GetFilterCount()`).
- `Height` is equal to `GetFilterHeight()`.
- `Width` is equal to `GetFilterWidth()`.
- `Depth` and `Channels` are equal to `1`.

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const;
```

The free terms are represented by a blob of the total size equal to the number of filters used (`GetFilterCount()`).

## Inputs

Each input accepts a blob with several images in RLE format. The dimensions of all inputs should be the same:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set.
- `Height` - the images' height.
- `Width` - the images' width, not more than `64`.
- `Depth` - the images' depth, should be equal to `1`.
- `Channels` - the number of channels, should be equal to `1`.

## Outputs

For each input the layer has one output. It contains a blob with the result of the convolution, in the same format as the regular [`CConvLayer`](ConvLayer.md). 

The output blob dimensions are:

- `BatchLength` is equal to the input `BatchLength`.
- `BatchWidth` is equal to the input `BatchWidth`.
- `ListSize` is equal to the input `ListSize`.
- `Height` can be calculated from the input `Height` as
`(Height - FilterHeight)/StrideHeight + 1`.
- `Width` can be calculated from the input `Width` as
`(Width - FilterWidth)/StrideWidth + 1`.
- `Depth` is equal to `1`.
- `Channels` is equal to `GetFilterCount()`.

