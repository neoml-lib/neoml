/* Copyright © 2017-2022 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

#include <TestParams.h>
#include <TestFixture.h>
#include <vector>

namespace NeoMLTest {

// Blob helpers

inline int getFlatIndex( const CFloatBlob& blob, int seq, int batch, int list, int channel, int depth, int row, int column )
{
	return ( list + blob.GetDesc().ListSize() * ( batch + blob.GetDesc().BatchWidth() * seq ) ) * blob.GetDesc().ObjectSize()
		+ channel + blob.GetDesc().Channels() * ( depth + blob.GetDesc().Depth() * ( column + row * blob.GetDesc().Width() ) );
}

// Matrix multiplication test helpers

inline void batchMultiplyMatrixByMatrixAndAddNaive( int batchSize, const std::vector<float>& first, const std::vector<float>& second,
	int firstHeight, int firstWidth, int secondWidth, std::vector<float>& result )
{
	const int firstMatrixSize = firstHeight * firstWidth;
	const int secondMatrixSize = firstWidth * secondWidth;
	const int resultMatrixSize = firstHeight * secondWidth;

	for( int b = 0; b < batchSize; ++b ) {
		for( int i = 0; i < firstHeight; ++i ) {
			for( int j = 0; j < secondWidth; ++j ) {
				for( int k = 0; k < firstWidth; ++k ) {
					result[b * resultMatrixSize + i * secondWidth + j] +=
						first[b * firstMatrixSize + i * firstWidth + k] * second[b * secondMatrixSize + k * secondWidth + j];
				}
			}
		}
	}
}

// 2d Conv test helpers

inline int calcConvOutputSize( int input, int padding, int filter, int dilation, int stride )
{
	return  1 + ( input - ( filter - 1 ) * dilation + 2 * padding - 1 ) / stride;
}

inline void batchConvolutionForward( const float* input, const float* filter, const float* freeTerms, float* output,
	int inputLength, int inputBatch, int inputHeight, int inputWidth, int inputDepth, int inputChannels,
	int paddingHeight, int paddingWidth, int filterCount, int filterHeight, int filterWidth,
	int dilationHeight, int dilationWidth, int strideHeight, int strideWidth )
{
	const int outputHeight = calcConvOutputSize( inputHeight, paddingHeight, filterHeight, dilationHeight, strideHeight );
	const int outputWidth = calcConvOutputSize( inputWidth, paddingWidth, filterWidth, dilationWidth, strideWidth );
	const int inputObjectSize = inputHeight * inputWidth * inputDepth * inputChannels;
	const int outputObjectSize = filterCount * outputHeight * outputWidth;
	const int channels = inputDepth * inputChannels;
	const int filterObjectSize = channels * filterHeight * filterWidth;

	for( int b = 0; b < inputLength * inputBatch; ++b ) {
		for( int h = 0; h < outputHeight; ++h ) {
			for( int w = 0; w < outputWidth; ++w ) {
				for( int outChannel = 0; outChannel < filterCount; ++outChannel ) {
					const int outputIndex = b * outputObjectSize + h * outputWidth * filterCount + w * filterCount + outChannel;
					output[outputIndex] = freeTerms[outChannel];

					for( int filterH = 0; filterH < filterHeight; ++filterH ) {
						for( int filterW = 0; filterW < filterWidth; ++filterW ) {
							for( int inChannel = 0; inChannel < channels; ++inChannel ) {
								const int inputH = h * strideHeight - paddingHeight + filterH * dilationHeight;
								const int inputW = w * strideWidth - paddingWidth + filterW * dilationWidth;

								if( inputH >= 0 && inputW >= 0 && inputH < inputHeight && inputW < inputWidth ) {
									const int inputIndex = b * inputObjectSize + inputH * inputWidth * channels + inputW * channels + inChannel;
									const int filterIndex = outChannel * filterObjectSize + filterH * filterWidth * channels + filterW * channels + inChannel;

									output[outputIndex] += filter[filterIndex] * input[inputIndex];
								}
							}
						}
					}
				}
			}
		}
	}
}

// 3d conv test helpers

struct CConv3dTestParams {
	int InputCount;
	int InputHeight;
	int InputWidth;
	int InputDepth;
	int InputChannels;

	int PaddingHeight;
	int PaddingWidth;
	int PaddingDepth;

	int FilterCount;
	int FilterHeight;
	int FilterWidth;
	int FilterDepth;

	int StrideHeight;
	int StrideWidth;
	int StrideDepth;

	CConv3dTestParams( int inputCount, int inputHeight, int inputWidth, int inputDepth, int inputChannels,
			int paddingHeight, int paddingWidth, int paddingDepth, int filterCount, int filterHeight, int filterWidth, int filterDepth,
			int strideHeight, int strideWidth, int strideDepth ) :
		InputCount( inputCount ),
		InputHeight( inputHeight ),
		InputWidth( inputWidth ),
		InputDepth( inputDepth ),
		InputChannels( inputChannels ),
		PaddingHeight( paddingHeight ),
		PaddingWidth( paddingWidth ),
		PaddingDepth( paddingDepth ),
		FilterCount( filterCount ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		FilterDepth( filterDepth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		StrideDepth( strideDepth )
	{
	}
};

inline CConv3dTestParams getConv3dParams( const CTestParams& params, CRandom& random )
{
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval inputDepthInterval = params.GetInterval( "InputDepth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval filterCountInterval = params.GetInterval( "FilterCount" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval filterDepthInterval = params.GetInterval( "FilterDepth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval strideDepthInterval = params.GetInterval( "StrideDepth" );
	const CInterval paddingHeightInterval = params.GetInterval( "PaddingHeight" );
	const CInterval paddingWidthInterval = params.GetInterval( "PaddingWidth" );
	const CInterval paddingDepthInterval = params.GetInterval( "PaddingDepth" );

	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputDepth = random.UniformInt( inputDepthInterval.Begin, inputDepthInterval.End );

	const int inputChannels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int filterCount = random.UniformInt( filterCountInterval.Begin, filterCountInterval.End );

	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int filterDepth = random.UniformInt( filterDepthInterval.Begin, filterDepthInterval.End );

	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );
	const int strideDepth = random.UniformInt( strideDepthInterval.Begin, strideDepthInterval.End );

	const int paddingHeight = random.UniformInt( paddingHeightInterval.Begin, paddingHeightInterval.End );
	const int paddingWidth = random.UniformInt( paddingWidthInterval.Begin, paddingWidthInterval.End );
	const int paddingDepth = random.UniformInt( paddingDepthInterval.Begin, paddingDepthInterval.End );

	return CConv3dTestParams( batchSize, inputHeight, inputWidth, inputDepth, inputChannels,
		paddingHeight, paddingWidth, paddingDepth,
		filterCount, filterHeight, filterWidth, filterDepth,
		strideHeight, strideWidth, strideDepth );
}

inline float* getInputElem( const CConv3dTestParams& params, int h, int w, int d, int ch, float* input )
{
	const int channels = params.InputChannels;

	if( h < 0 || w < 0 || d < 0 || h >= params.InputHeight || w >= params.InputWidth || d >= params.InputDepth ) {
		return 0;
	}

	return input + ( h * params.InputWidth * params.InputDepth * channels + w * params.InputDepth * channels + d * channels + ch );
}

// 2d pooling test helpers

struct CPoolingTestParams {
	int InputCount;
	int InputHeight;
	int InputWidth;
	int InputDepth;
	int InputChannels;

	int FilterHeight;
	int FilterWidth;

	int StrideHeight;
	int StrideWidth;

	int OutputHeight;
	int OutputWidth;

	bool IsMaxPooing = false;

	CPoolingTestParams( int inputCount, int inputHeight, int inputWidth, int inputDepth, int inputChannels,
			int filterHeight, int filterWidth, int strideHeight, int strideWidth, int outputHeight, int outputWidth ) :
		InputCount( inputCount ),
		InputHeight( inputHeight ),
		InputWidth( inputWidth ),
		InputDepth( inputDepth ),
		InputChannels( inputChannels ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		OutputHeight( outputHeight ),
		OutputWidth( outputWidth )
	{}
};

inline CPoolingTestParams getPoolingParams( const CTestParams& params, CRandom& random )
{
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval inputDepthInterval = params.GetInterval( "InputDepth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );

	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputDepth = random.UniformInt( inputDepthInterval.Begin, inputDepthInterval.End );

	const int inputChannels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );

	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );

	const int outHeight = calcConvOutputSize( inputHeight, 0, filterHeight, 1, strideHeight );
	const int outWidth = calcConvOutputSize( inputWidth, 0, filterWidth, 1, strideWidth );

	return CPoolingTestParams( batchSize, inputHeight, inputWidth, inputDepth, inputChannels,
		filterHeight, filterWidth, strideHeight, strideWidth, outHeight, outWidth );
}

// 3d pooling test helpers

struct C3dPoolingTestParams {
	int InputCount;
	int InputHeight;
	int InputWidth;
	int InputDepth;
	int InputChannels;

	int FilterHeight;
	int FilterWidth;
	int FilterDepth;

	int StrideHeight;
	int StrideWidth;
	int StrideDepth;

	bool IsMaxPooing = false;

	C3dPoolingTestParams( int inputCount, int inputHeight, int inputWidth, int inputDepth, int inputChannels,
			int filterHeight, int filterWidth, int filterDepth, int strideHeight, int strideWidth, int strideDepth ) :
		InputCount( inputCount ),
		InputHeight( inputHeight ),
		InputWidth( inputWidth ),
		InputDepth( inputDepth ),
		InputChannels( inputChannels ),
		FilterHeight( filterHeight ),
		FilterWidth( filterWidth ),
		FilterDepth( filterDepth ),
		StrideHeight( strideHeight ),
		StrideWidth( strideWidth ),
		StrideDepth( strideDepth )
	{}
};

inline C3dPoolingTestParams get3dPoolingParams( const CTestParams& params, CRandom& random )
{
	const CInterval inputHeightInterval = params.GetInterval( "InputHeight" );
	const CInterval inputWidthInterval = params.GetInterval( "InputWidth" );
	const CInterval inputDepthInterval = params.GetInterval( "InputDepth" );
	const CInterval channelsInterval = params.GetInterval( "Channels" );
	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval filterHeightInterval = params.GetInterval( "FilterHeight" );
	const CInterval filterWidthInterval = params.GetInterval( "FilterWidth" );
	const CInterval filterDepthInterval = params.GetInterval( "FilterDepth" );
	const CInterval strideHeightInterval = params.GetInterval( "StrideHeight" );
	const CInterval strideWidthInterval = params.GetInterval( "StrideWidth" );
	const CInterval strideDepthInterval = params.GetInterval( "StrideDepth" );

	const int inputHeight = random.UniformInt( inputHeightInterval.Begin, inputHeightInterval.End );
	const int inputWidth = random.UniformInt( inputWidthInterval.Begin, inputWidthInterval.End );
	const int inputDepth = random.UniformInt( inputDepthInterval.Begin, inputDepthInterval.End );

	const int inputChannels = random.UniformInt( channelsInterval.Begin, channelsInterval.End );
	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );

	const int filterHeight = random.UniformInt( filterHeightInterval.Begin, filterHeightInterval.End );
	const int filterWidth = random.UniformInt( filterWidthInterval.Begin, filterWidthInterval.End );
	const int filterDepth = random.UniformInt( filterDepthInterval.Begin, filterDepthInterval.End );

	const int strideHeight = random.UniformInt( strideHeightInterval.Begin, strideHeightInterval.End );
	const int strideWidth = random.UniformInt( strideWidthInterval.Begin, strideWidthInterval.End );
	const int strideDepth = random.UniformInt( strideDepthInterval.Begin, strideDepthInterval.End );

	return C3dPoolingTestParams( batchSize, inputHeight, inputWidth, inputDepth, inputChannels,
		filterHeight, filterWidth, filterDepth, strideHeight, strideWidth, strideDepth );
}

// Softmax test helpers

inline void softmaxImpl( const std::vector<float>& input, int height, int width, bool byRow, std::vector<float>& output )
{
	std::vector<float> maxima;
	maxima.resize( byRow ? height : width );
	output = input;

	for( int i = 0; i < height; ++i ) {
		for( int j = 0; j < width; ++j ) {
			if( byRow ) {
				maxima[i] = j == 0 ? input[i * width + j] : std::max( maxima[i], input[i * width + j] );
			}
			else {
				maxima[j] = i == 0 ? input[i * width + j] : std::max( maxima[j], input[i * width + j] );
			}
		}
	}

	for( int i = 0; i < height; ++i ) {
		for( int j = 0; j < width; ++j ) {
			output[i * width + j] -= byRow ? maxima[i] : maxima[j];
			output[i * width + j] = expf( std::min( FLT_MAX_LOG, std::max( FLT_MIN_LOG, output[i * width + j] ) ) );
		}
	}

	std::vector<float> sums;
	sums.resize( byRow ? height : width );

	for( int i = 0; i < height; ++i ) {
		for( int j = 0; j < width; ++j ) {
			if( byRow ) {
				sums[i] = j == 0 ? output[i * width + j] : sums[i] + output[i * width + j];
			}
			else {
				sums[j] = i == 0 ? output[i * width + j] : sums[j] + output[i * width + j];
			}
		}
	}

	for( int i = 0; i < height; ++i ) {
		for( int j = 0; j < width; ++j ) {
			output[i * width + j] /= byRow ? sums[i] : sums[j];
		}
	}
}

// IndRnn test helpers

inline float sigmoidDiffOp( float output, float outputDiff )
{
	return outputDiff * output * ( 1.f - output );
}

inline float reluDiffOp( float output, float outputDiff )
{
	return output > 0.f ? outputDiff : 0.f;
}

typedef float( *TTestActivationDiffOp ) ( float output, float outputDiff );

// AddHeight/WidthIndex helpers

inline void addIndexNaive( const int* input, int batchSize, int height, int width, int channels, int* output, bool isAddHeight )
{
	for( int batch = 0; batch < batchSize; ++batch ) {
		for( int h = 0; h < height; ++h ) {
			for( int w = 0; w < width; ++w ) {
				for( int c = 0; c < channels; ++c ) {
					*output = *input + ( isAddHeight ? h : w );
					input++;
					output++;
				}
			}
		}
	}
}

// Randomization test helpers

template<int size>
class CIntArray {
public:
	static const int Size = size;

	CIntArray();

	const unsigned int& operator[] (int index) const { return data[index]; }
	unsigned int& operator[] (int index) { return data[index]; }

	const unsigned int* GetPtr() const { return data; }

private:
	unsigned int data[size];
};

template<int size>
inline CIntArray<size>::CIntArray()
{
	for (int i = 0; i < size; ++i) {
		data[i] = 0;
	}
}

class CExpectedRandom {
public:
	explicit CExpectedRandom( int seed );

	void Skip( unsigned long long count );

	CIntArray<4> Next();

private:
	static const unsigned int kPhiloxW32A = 0x9E3779B9;
	static const unsigned int kPhiloxW32B = 0xBB67AE85;
	static const unsigned int kPhiloxM4x32A = 0xD2511F53;
	static const unsigned int kPhiloxM4x32B = 0xCD9E8D57;

	CIntArray<4> counter;
	CIntArray<2> key;

	static void raiseKey(CIntArray<2>& key);
	static CIntArray<4> computeSingleRound(const CIntArray<4>& counter, const CIntArray<2>& key);
	void skipOne();
};

inline CExpectedRandom::CExpectedRandom(int seed)
{
	key[0] = seed;
	key[1] = seed ^ 0xBADF00D;
	counter[2] = seed ^ 0xBADFACE;
	counter[3] = seed ^ 0xBADBEEF;
}

inline void CExpectedRandom::Skip(unsigned long long count)
{
	const unsigned int countLow = static_cast<unsigned int>(count);
	unsigned int countHigh = static_cast<unsigned int>(count >> 32);

	counter[0] += countLow;
	if (counter[0] < countLow) {
		countHigh++;
	}

	counter[1] += countHigh;
	if (counter[1] < countHigh) {
		if (++counter[2] == 0) {
			++counter[3];
		}
	}
}

inline CIntArray<4> CExpectedRandom::Next()
{
	CIntArray<4> currentCounter = counter;
	CIntArray<2> currentKey = key;

	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);
	currentCounter = computeSingleRound(currentCounter, currentKey);
	raiseKey(currentKey);

	skipOne();

	return currentCounter;
}

inline void CExpectedRandom::raiseKey(CIntArray<2>& key)
{
	key[0] += kPhiloxW32A;
	key[1] += kPhiloxW32B;
}

inline void multiplyHighLow(unsigned int x, unsigned int y, unsigned int* resultLow,
	unsigned int* resultHigh)
{
	const unsigned long long product = static_cast<unsigned long long>(x) * y;
	*resultLow = static_cast<unsigned int>(product);
	*resultHigh = static_cast<unsigned int>(product >> 32);
}

inline CIntArray<4> CExpectedRandom::computeSingleRound(const CIntArray<4>& counter, const CIntArray<2>& key)
{
	unsigned int firstLow;
	unsigned int firstHigh;
	multiplyHighLow(kPhiloxM4x32A, counter[0], &firstLow, &firstHigh);

	unsigned int secondLow;
	unsigned int secondHigh;
	multiplyHighLow(kPhiloxM4x32B, counter[2], &secondLow, &secondHigh);

	CIntArray<4> result;
	result[0] = secondHigh ^ counter[1] ^ key[0];
	result[1] = secondLow;
	result[2] = firstHigh ^ counter[3] ^ key[1];
	result[3] = firstLow;
	return result;
}

inline void CExpectedRandom::skipOne()
{
	if (++counter[0] == 0) {
		if (++counter[1] == 0) {
			if (++counter[2] == 0) {
				++counter[3];
			}
		}
	}
}

} // namespace NeoMLTest
