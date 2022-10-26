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

#include "common.h"
#pragma hdrstop

#include <stdio.h>
#include <gtest/gtest.h>

#include <TestParams.h>

using namespace NeoML;

namespace NeoMLTest {

// The class to generate random values
// It uses the complementary-multiply-with-carry algorithm
// C lag-1024, multiplier(a) = 108798, initial carry(c) = 12345678
class CRandom {
public:
	explicit CRandom( unsigned int seed = 0xBADF00D ) { srand(seed); }

	// Returns the next random value
	unsigned int Next() { return (rand() << 16) + rand(); }

	// Returns a double value from a uniform distribution in [ min, max ) range
	// If min == max, min is returned
	double Uniform( double min, double max ) { return min + (max - min) * Next() / 4294967296.; }

	// Returns an int value from a uniform distribution in [ min, max ] range. Note that max return value is possible!
	// If min == max, min is returned
	int UniformInt( int min, int max ) { return (int)(min + (unsigned int)(((static_cast<unsigned long long>(max) - min + 1) * Next()) >> 32));  }
};

template<class T>
class CBufferWrapper {
public:
	CBufferWrapper( IMathEngine& _mathEngine, T* _data, int _size) : mathEngine( _mathEngine ), isCopyBack(false), size(_size), data(_data)
	{
		mathData = CTypedMemoryHandle<T>( mathEngine.HeapAlloc( size * sizeof(T) ) );
		mathEngine.DataExchangeTyped<T>(CTypedMemoryHandle<T>(mathData), data, size);
	}

	~CBufferWrapper()
	{
		if(isCopyBack) {
			mathEngine.DataExchangeTyped<T>(data, CTypedMemoryHandle<T>(mathData), size);
		}
		mathEngine.HeapFree(mathData);
	}

	operator CTypedMemoryHandle<T>() const { isCopyBack = true; return CTypedMemoryHandle<T>(mathData); }
	operator CTypedMemoryHandle<const T>() const { return CTypedMemoryHandle<const T>(mathData); }

private:
	IMathEngine& mathEngine;
	mutable bool isCopyBack;
	int size;
	T* data;
	CTypedMemoryHandle<T> mathData;
};

typedef CBufferWrapper<float> CFloatWrapper;
typedef CBufferWrapper<int> CIntWrapper;

#define CARRAY_WRAPPER(TYPE, arr) CBufferWrapper<TYPE>( MathEngine(), ( arr.data() ), ( static_cast<int>( arr.size() ) ) )
#define CARRAY_FLOAT_WRAPPER(arr) CARRAY_WRAPPER(float, arr)
#define CARRAY_INT_WRAPPER(arr) CARRAY_WRAPPER(int, arr)

#define ARR_WRAPPER(TYPE, arr) CBufferWrapper<TYPE>( MathEngine(), (arr), (int)sizeof(arr) / sizeof(TYPE) )
#define FLOAT_WRAPPER(arr) CFloatWrapper( MathEngine(), (arr), (int)sizeof(arr) / sizeof(float) )
#define FLOAT_WRAPPER_MATHENGINE(mathEngine, arr) CFloatWrapper( mathEngine, (arr), (int)sizeof(arr) / sizeof(float) )
#define INT_WRAPPER(arr) CIntWrapper( MathEngine(), (arr), (int)sizeof(arr) / sizeof(int) )
#define INT_WRAPPER_MATHENGINE(mathEngine, arr) CIntWrapper( mathEngine, (arr), (int)sizeof(arr) / sizeof(int) )

#define CREATE_FILL_ARRAY(TYPE, arr, min, max, size, random) \
	std::vector<TYPE> arr; \
	arr.resize( size ); \
	for(int i = 0; i < size; ++i) { \
		arr[i] = static_cast<TYPE>( random.Uniform( min, max ) ); \
	}

#define CREATE_FILL_FLOAT_ARRAY(arr, min, max, size, random) \
	CREATE_FILL_ARRAY(float, arr, min, max, size, random)

// Leaving CREATE_FILL_INT_ARRAY as it was before CREATE_FILL_ARRAY
// for the sake of backward compatibility
#define CREATE_FILL_INT_ARRAY(arr, min, max, size, random) \
	std::vector<int> arr; \
	arr.resize( size ); \
	for(int i = 0; i < size; ++i) { \
		arr[i] = random.UniformInt( min, max ); \
	}

static inline int calcConvOutputSize( int input, int padding, int filter, int dilation, int stride )
{
	return  1 + ( input - ( filter - 1 ) * dilation + 2 * padding - 1 ) / stride;
}

} // namespace NeoMLTest

using namespace NeoMLTest;

class CBlockedConvTest : public ::testing::Test, public ::testing::WithParamInterface<CTestParams> {
};

static const int RUN_COUNT = 100;

TEST_P( CBlockedConvTest, Ideal )
{
	const CTestParams& params = GetParam();
	CRandom random( params.GetValue<int>( "Seed" ) );
	const int batch = params.GetValue<int>( "Batch" );
	const int height = params.GetValue<int>( "Height" );
	const int width = params.GetValue<int>( "Width" );
	const int channels = params.GetValue<int>( "Channels" );
	const int filterCount = params.GetValue<int>( "FilterCount" );
	const int filterHeight = params.GetValue<int>( "FilterHeight" );
	const int filterWidth = params.GetValue<int>( "FilterWidth" );
	const int strideHeight = params.GetValue<int>( "StrideHeight" );
	const int strideWidth = params.GetValue<int>( "StrideWidth" );
	const int paddingHeight = params.GetValue<int>( "PaddingHeight" );
	const int paddingWidth = params.GetValue<int>( "PaddingWidth" );
	const int dilationHeight = params.GetValue<int>( "DilationHeight" );
	const int dilationWidth = params.GetValue<int>( "DilationWidth" );
	if( filterHeight == 1 && filterWidth == 1 ) {
		if( strideHeight == 1 && strideWidth == 1 && paddingHeight == 0 && paddingWidth == 0 ) {
			GTEST_SKIP() << "We cannot beat MKL at this...";
		} else {
			GTEST_FAIL() << "We cannot beat MKL at this...";
		}
	}

	const int outputHeight = calcConvOutputSize( height, paddingHeight, filterHeight, dilationHeight, strideHeight );
	const int outputWidth = calcConvOutputSize( width, paddingWidth, filterWidth, dilationWidth, strideWidth );

	CREATE_FILL_FLOAT_ARRAY( neomlInput, -2.f, 2.f, batch * height * width * channels, random );
	CBlobDesc inputDesc( { 1, batch, 1, height, width, 1, channels } );

	CREATE_FILL_FLOAT_ARRAY( neomlFilter, -2.f, 2.f, filterCount * filterHeight * filterWidth * channels, random );
	CBlobDesc filterDesc( { 1, filterCount, 1, filterHeight, filterWidth, 1, channels } );

	CREATE_FILL_FLOAT_ARRAY( bias, -5.f, 5.f, filterCount, random );

	CBlobDesc outputDesc( { 1, batch, 1, outputHeight, outputWidth, 1, filterCount } );
	std::vector<float> expectedOutput( batch * outputHeight * outputWidth * filterCount );
	
	std::unique_ptr<IPerformanceCounters> counters( MathEngine().CreatePerformanceCounters() );

	using CPerfStat = std::vector<IPerformanceCounters::CCounter::TCounterType>;

	auto update = []( const IPerformanceCounters& counters, CPerfStat& stat )
	{
		for( size_t i = 0; i < counters.size(); ++i ) {
			stat[i] += counters[i].Value;
		}
	};

	CPerfStat neomlPerf( counters->size() );
	CPerfStat inputConversionPerf( counters->size() );
	CPerfStat blockedConvPerf( counters->size() );
	CPerfStat outputConversionPerf( counters->size() );

	{
		CConvolutionDesc* convDesc = MathEngine().InitBlobConvolution( inputDesc, paddingHeight, paddingWidth, strideHeight,
			strideWidth, dilationHeight, dilationWidth, filterDesc, outputDesc );
		auto biasHandle = CARRAY_FLOAT_WRAPPER( bias );
		auto inputHandle = CARRAY_FLOAT_WRAPPER( neomlInput );
		auto filterHandle = CARRAY_FLOAT_WRAPPER( neomlFilter );
		auto outputHandle = CARRAY_FLOAT_WRAPPER( expectedOutput );
		MathEngine().BlobConvolution( *convDesc, inputHandle, filterHandle, &static_cast<CConstFloatHandle>( biasHandle ), outputHandle );
		for( int run = 0; run < RUN_COUNT; ++run ) {
			counters->Synchronise();
			MathEngine().BlobConvolution( *convDesc, inputHandle, filterHandle, &static_cast<CConstFloatHandle>( biasHandle ), outputHandle );
			counters->Synchronise();
			update( *counters, neomlPerf );
		}
		delete convDesc;
	}

	std::vector<float> blockedInput( neomlInput.size() );
	SimdMathEngine().PackBlockedData( inputDesc, neomlInput.data(), blockedInput.data() );
	for( int run = 0; run < RUN_COUNT; ++run ) {
		counters->Synchronise();
		SimdMathEngine().PackBlockedData( inputDesc, neomlInput.data(), blockedInput.data() );
		counters->Synchronise();
		update( *counters, inputConversionPerf );
	}

	std::vector<float> blockedFilter( neomlFilter.size() );
	SimdMathEngine().PackBlockedFilter( filterDesc, neomlFilter.data(), blockedFilter.data() );
	std::vector<float> blockedOutput( expectedOutput.size() );

	CConvolutionDesc* simdConvDesc = SimdMathEngine().InitBlockedConvolution( inputDesc, paddingHeight, paddingWidth,
		strideHeight, strideWidth, dilationHeight, dilationWidth, filterDesc, outputDesc );
	ASSERT_TRUE( simdConvDesc != nullptr );
	SimdMathEngine().BlockedConvolution( *simdConvDesc, blockedInput.data(), blockedFilter.data(),
		bias.data(), blockedOutput.data() );
	for( int run = 0; run < RUN_COUNT; ++run ) {
		counters->Synchronise();
		SimdMathEngine().BlockedConvolution( *simdConvDesc, blockedInput.data(), blockedFilter.data(),
			bias.data(), blockedOutput.data() );
		counters->Synchronise();
		update( *counters, blockedConvPerf );
	}
	delete simdConvDesc;

	std::vector<float> actualOutput( blockedOutput.size() );
	SimdMathEngine().UnpackBlockedData( outputDesc, blockedOutput.data(), actualOutput.data() );
	for( int run = 0; run < RUN_COUNT; ++run ) {
		counters->Synchronise();
		SimdMathEngine().UnpackBlockedData( outputDesc, blockedOutput.data(), actualOutput.data() );
		counters->Synchronise();
		update( *counters, outputConversionPerf );
	}

	for( size_t i = 0; i < actualOutput.size(); ++i ) {
		if( ::fabsf( actualOutput[i] - expectedOutput[i] ) > 1e-2f ) {
			//__debugbreak();
		}
		ASSERT_NEAR( actualOutput[i], expectedOutput[i], 1e-2f ) << "at #" << i;
	}

	std::cout << "NeoML\tInput\tRunCount\tOutput\tConvRatio\tFullRatio\n"
		<< neomlPerf[0] / 1e9 << '\t'
		<< inputConversionPerf[0] / 1e9 << '\t'
		<< blockedConvPerf[0] / 1e9 << '\t'
		<< outputConversionPerf[0] / 1e9 << '\t'
		<< 100. * blockedConvPerf[0] / neomlPerf[0] << "%\t"
		<< 100. * ( inputConversionPerf[0] + blockedConvPerf[0] + outputConversionPerf[0] ) / neomlPerf[0] << "%\n";
}

TEST_P( CBlockedConvTest, Real )
{
	const CTestParams& params = GetParam();
	CRandom random( params.GetValue<int>( "Seed" ) );
	const int batch = params.GetValue<int>( "Batch" );
	const int height = params.GetValue<int>( "Height" );
	const int width = params.GetValue<int>( "Width" );
	const int channels = params.GetValue<int>( "Channels" );
	const int filterCount = params.GetValue<int>( "FilterCount" );
	const int filterHeight = params.GetValue<int>( "FilterHeight" );
	const int filterWidth = params.GetValue<int>( "FilterWidth" );
	const int strideHeight = params.GetValue<int>( "StrideHeight" );
	const int strideWidth = params.GetValue<int>( "StrideWidth" );
	const int paddingHeight = params.GetValue<int>( "PaddingHeight" );
	const int paddingWidth = params.GetValue<int>( "PaddingWidth" );
	const int dilationHeight = params.GetValue<int>( "DilationHeight" );
	const int dilationWidth = params.GetValue<int>( "DilationWidth" );
	if( filterHeight == 1 && filterWidth == 1 ) {
		GTEST_SKIP() << "We cannot beat MKL at this...";
	}

	const int outputHeight = calcConvOutputSize( height, paddingHeight, filterHeight, dilationHeight, strideHeight );
	const int outputWidth = calcConvOutputSize( width, paddingWidth, filterWidth, dilationWidth, strideWidth );

	CREATE_FILL_FLOAT_ARRAY( neomlInput, -2.f, 2.f, batch * height * width * channels, random );
	CBlobDesc inputDesc( { 1, batch, 1, height, width, 1, channels } );

	CREATE_FILL_FLOAT_ARRAY( neomlFilter, -2.f, 2.f, filterCount * filterHeight * filterWidth * channels, random );
	CBlobDesc filterDesc( { 1, filterCount, 1, filterHeight, filterWidth, 1, channels } );

	CREATE_FILL_FLOAT_ARRAY( bias, -5.f, 5.f, filterCount, random );

	CBlobDesc outputDesc( { 1, batch, 1, outputHeight, outputWidth, 1, filterCount } );
	std::vector<float> expectedOutput( batch * outputHeight * outputWidth * filterCount );

	std::unique_ptr<IPerformanceCounters> counters( MathEngine().CreatePerformanceCounters() );

	using CPerfStat = std::vector<IPerformanceCounters::CCounter::TCounterType>;

	auto update = []( const IPerformanceCounters& counters, CPerfStat& stat )
	{
		for( size_t i = 0; i < counters.size(); ++i ) {
			stat[i] += counters[i].Value;
		}
	};

	CPerfStat neomlPerf( counters->size() );
	CPerfStat allocPerf( counters->size() );
	CPerfStat inputConversionPerf( counters->size() );
	CPerfStat filterConversionPerf( counters->size() );
	CPerfStat blockedConvPerf( counters->size() );
	CPerfStat outputConversionPerf( counters->size() );
	CPerfStat freePerf( counters->size() );

	{
		CConvolutionDesc* convDesc = MathEngine().InitBlobConvolution( inputDesc, paddingHeight, paddingWidth, strideHeight,
			strideWidth, dilationHeight, dilationWidth, filterDesc, outputDesc );
		auto biasHandle = CARRAY_FLOAT_WRAPPER( bias );
		auto inputHandle = CARRAY_FLOAT_WRAPPER( neomlInput );
		auto filterHandle = CARRAY_FLOAT_WRAPPER( neomlFilter );
		auto outputHandle = CARRAY_FLOAT_WRAPPER( expectedOutput );
		for( int run = 0; run <= RUN_COUNT; ++run ) {
			counters->Synchronise();
			MathEngine().BlobConvolution( *convDesc, inputHandle, filterHandle, &static_cast<CConstFloatHandle>( biasHandle ), outputHandle );
			counters->Synchronise();
			if( run != 0 ) {
				update( *counters, neomlPerf );
			}
		}
		delete convDesc;
	}

	for( int run = 0; run <= RUN_COUNT; ++run ) {
		CFloatHandleStackVar* stackVar;
		counters->Synchronise();
		stackVar = new CFloatHandleStackVar( MathEngine(), neomlInput.size() + neomlFilter.size() + expectedOutput.size() );
		float* blockedInput = static_cast<float*>( MathEngine().GetBuffer( stackVar->GetHandle(), 0, stackVar->Size(), false ) );
		float* blockedFilter = blockedInput + neomlInput.size();
		float* blockedOutput = blockedFilter + neomlFilter.size();
		counters->Synchronise();
		if( run != 0 ) update( *counters, allocPerf );

		counters->Synchronise();
		SimdMathEngine().PackBlockedData( inputDesc, neomlInput.data(), blockedInput );
		counters->Synchronise();
		if( run != 0 ) update( *counters, inputConversionPerf );

		counters->Synchronise();
		SimdMathEngine().PackBlockedFilter( filterDesc, neomlFilter.data(), blockedFilter );
		counters->Synchronise();
		if( run != 0 ) update( *counters, filterConversionPerf );

		CConvolutionDesc* simdConvDesc = SimdMathEngine().InitBlockedConvolution( inputDesc, paddingHeight, paddingWidth,
			strideHeight, strideWidth, dilationHeight, dilationWidth, filterDesc, outputDesc );
		ASSERT_TRUE( simdConvDesc != nullptr );
		counters->Synchronise();
		SimdMathEngine().BlockedConvolution( *simdConvDesc, blockedInput, blockedFilter, bias.data(), blockedOutput );
		counters->Synchronise();
		if( run != 0 ) update( *counters, blockedConvPerf );
		delete simdConvDesc;

		std::vector<float> actualOutput( expectedOutput.size() );
		counters->Synchronise();
		SimdMathEngine().UnpackBlockedData( outputDesc, blockedOutput, actualOutput.data() );
		counters->Synchronise();
		if( run != 0 ) update( *counters, outputConversionPerf );

		counters->Synchronise();
		delete stackVar;
		counters->Synchronise();
		if( run != 0 ) update( *counters, freePerf );

		if( run == 0 ) {
			for( size_t i = 0; i < actualOutput.size(); ++i ) {
				if( ::fabsf( actualOutput[i] - expectedOutput[i] ) > 1e-2f ) {
					//__debugbreak();
				}
				ASSERT_NEAR( actualOutput[i], expectedOutput[i], 1e-2f ) << "at #" << i;
			}
		}
	}

	std::cout << "NeoML\tAlloc\tInput\tFilter\tBlocked\tOutput\tFree\tInputSize\tOutputSize\tTotalOperations\tCoeffI\tCoeffO\tCoeffF\n"
		<< neomlPerf[0] / 1e9 << '\t'
		<< allocPerf[0] / 1e9 << '\t'
		<< inputConversionPerf[0] / 1e9 << '\t'
		<< filterConversionPerf[0] / 1e9 << '\t'
		<< blockedConvPerf[0] / 1e9 << '\t'
		<< outputConversionPerf[0] / 1e9 << '\t'
		<< freePerf[0] / 1e9 << '\t'
		<< inputDesc.BlobSize() << '\t'
		<< outputDesc.BlobSize() << '\t'
		<< outputDesc.BlobSize() * channels * filterHeight * filterWidth << '\t'
		<< outputDesc.Height() * outputDesc.Width() * filterCount * filterHeight * filterWidth / ( height * width ) << '\t'
		<< channels * filterHeight * filterWidth << '\t'
		<< batch * outputDesc.Height() * outputDesc.Width()
		<< '\n';
}

struct BlockedConvTestNameGenerator {
	std::string operator()(const testing::TestParamInfo<CTestParams>& paramInfo)
	{
		const CTestParams& params = paramInfo.param;
		if( !params.Name().empty() ) {
			return params.Name();
		}

		const int batch = params.GetValue<int>( "Batch" );
		const int height = params.GetValue<int>( "Height" );
		const int width = params.GetValue<int>( "Width" );
		const int channels = params.GetValue<int>( "Channels" );
		const int filterCount = params.GetValue<int>( "FilterCount" );
		const int filterHeight = params.GetValue<int>( "FilterHeight" );
		const int filterWidth = params.GetValue<int>( "FilterWidth" );
		const int strideHeight = params.GetValue<int>( "StrideHeight" );
		const int strideWidth = params.GetValue<int>( "StrideWidth" );
		const int paddingHeight = params.GetValue<int>( "PaddingHeight" );
		const int paddingWidth = params.GetValue<int>( "PaddingWidth" );
		const int dilationHeight = params.GetValue<int>( "DilationHeight" );
		const int dilationWidth = params.GetValue<int>( "DilationWidth" );


		const int bufferSize = 1024;
		int currLen = 0;
		std::vector<char> buffer;
		buffer.resize( 1024 );

		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "I_%dx%dx%dx%d__", batch, height, width, channels );
		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "F_%dx%dx%d__", filterCount, filterHeight, filterWidth );
		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "St_%dx%d__", strideHeight, strideWidth );
		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "P_%dx%d__", paddingHeight, paddingWidth );
		currLen += ::sprintf_s( buffer.data() + currLen, bufferSize - currLen, "D_%dx%d", dilationHeight, dilationWidth );

		return std::string( buffer.data(), currLen );
	}
};

INSTANTIATE_TEST_SUITE_P( Trivial, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Minimal"
		),
		CTestParams(
			"Batch = 2;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Batch"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 2;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"InputHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 2;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"InputWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 16;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"InputChannels"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"FilterCount"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 2;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 2;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"FilterHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 2;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 2;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"FilterWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 2;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 3;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"PaddingHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 3;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"TrickyPaddingHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 2;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"PaddingWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"TrickyPaddingWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 3;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 2;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"StrideHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 3;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 2;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"StrideWidth"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 3;"
			"Width = 1;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 2;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 2;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"DilationHeight"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 1;"
			"Width = 3;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 1;"
			"FilterWidth = 2;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 2;"
			"Seed = 348;",
			"DilationWidth"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P(Yolox0, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 320;"
			"Width = 640;"
			"Channels = 32;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_16_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 64;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_19_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_25_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_28_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 64;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_22_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_33_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 160;"
			"Width = 320;"
			"Channels = 64;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_36_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_39_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_45_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_48_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_52_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_55_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_59_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_62_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_42_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_67_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_70_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_73_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_79_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_82_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_86_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_89_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_93_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_96_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_76_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_101_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 512;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_104_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_107_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 1024;"
			"FilterCount = 512;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_114_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_117_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_123_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_126_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_120_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 512;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_130_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_133_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 512;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_138_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_144_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_147_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 512;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_141_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_151_Op"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P(Yolox1, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_154_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 256;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_159_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_165_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_168_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 256;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_162_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_172_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_215_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_218_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_221_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_175_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_179_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_185_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_188_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_182_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_192_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_233_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_236_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_239_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_195_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_199_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_205_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_208_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_202_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 512;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_212_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 512;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_251_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_254_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_257_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_224_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 80;"
			"Width = 160;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_227_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_242_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 40;"
			"Width = 80;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_245_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_260_Op"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 20;"
			"Width = 40;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 348;",
			"Conv_263_Op"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P(DANet, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 36;"
			"Width = 24;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"26"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 72;"
			"Width = 48;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 9;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 4;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"16"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 72;"
			"Width = 48;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 9;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"14"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 288;"
			"Width = 192;"
			"Channels = 8;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"8"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 144;"
			"Width = 96;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"11"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 72;"
			"Width = 48;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"18"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 36;"
			"Width = 24;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 9;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 4;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"24"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 36;"
			"Width = 24;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 9;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"22"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 288;"
			"Width = 192;"
			"Channels = 16;"
			"FilterCount = 8;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"49"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 18;"
			"Width = 12;"
			"Channels = 64;"
			"FilterCount = 32;"
			"FilterHeight = 9;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"30"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 18;"
			"Width = 12;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 9;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 4;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"32"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 18;"
			"Width = 12;"
			"Channels = 64;"
			"FilterCount = 32;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"34"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 288;"
			"Width = 192;"
			"Channels = 8;"
			"FilterCount = 8;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"46"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( BusinessObjects, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 288;"
			"Width = 192;"
			"Channels = 8;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"8"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 144;"
			"Width = 96;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 8;"
			"PaddingWidth = 8;"
			"DilationHeight = 8;"
			"DilationWidth = 8;"
			"Seed = 1984;",
			"conv_10"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 288;"
			"Width = 192;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_6"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 144;"
			"Width = 96;"
			"Channels = 16;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_7"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 144;"
			"Width = 96;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 2;"
			"Seed = 1984;",
			"conv_8"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 144;"
			"Width = 96;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 4;"
			"DilationHeight = 4;"
			"DilationWidth = 4;"
			"Seed = 1984;",
			"conv_9"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 144;"
			"Width = 96;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 16;"
			"PaddingWidth = 16;"
			"DilationHeight = 16;"
			"DilationWidth = 16;"
			"Seed = 1984;",
			"conv11"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 144;"
			"Width = 96;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv_final"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( KoreanTwoStageRleSecondLevel, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 12;"
			"Channels = 32;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv0"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 12;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 10;"
			"Width = 10;"
			"Channels = 16;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv2"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( EndToEndArabicFast, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 64;"
			"Channels = 160;"
			"FilterCount = 80;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( EndToEndArabic, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 32;"
			"Channels = 160;"
			"FilterCount = 80;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv0"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 32;"
			"Channels = 80;"
			"FilterCount = 160;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv1"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( JapaneseTwoStageGrayFirstLevel, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 12;"
			"Channels = 16;"
			"FilterCount = 8;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 3;"
			"StrideWidth = 3;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv0"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 8;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv1"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( JapaneseTwoStageGraySecondLevel, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 15;"
			"Width = 15;"
			"Channels = 8;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv0"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 15;"
			"Width = 15;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 3;"
			"StrideWidth = 3;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 5;"
			"Width = 5;"
			"Channels = 16;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv2"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( SegLinks, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block1_conv2"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 64;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block2_conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 128;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv_9_2"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block2_conv2"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 128;"
			"Width = 128;"
			"Channels = 128;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block3_conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 128;"
			"Width = 128;"
			"Channels = 256;"
			"FilterCount = 256;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block3_conv2"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 64;"
			"Width = 64;"
			"Channels = 256;"
			"FilterCount = 512;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block4_conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 64;"
			"Width = 64;"
			"Channels = 512;"
			"FilterCount = 512;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block4_conv2"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 512;"
			"FilterCount = 512;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block5_conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 512;"
			"FilterCount = 1024;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv_6"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 1024;"
			"FilterCount = 1024;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv_7"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 1024;"
			"FilterCount = 256;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv_8_1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 256;"
			"FilterCount = 512;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv_8_2"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 512;"
			"FilterCount = 128;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv_9_1"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( SegLinksDepthwise, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 64;"
			"Width = 64;"
			"Channels = 24;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_4_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"expanded_conv_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 16;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_1_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 128;"
			"Width = 128;"
			"Channels = 96;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_1_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 64;"
			"Width = 64;"
			"Channels = 144;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_3_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 128;"
			"Width = 128;"
			"Channels = 24;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_2_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 128;"
			"Width = 128;"
			"Channels = 144;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_2_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 144;"
			"FilterCount = 48;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_6_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 48;"
			"FilterCount = 288;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_7_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 288;"
			"FilterCount = 48;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_7_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 288;"
			"FilterCount = 72;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_10_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 72;"
			"FilterCount = 432;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_11_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 432;"
			"FilterCount = 72;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_11_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 432;"
			"FilterCount = 104;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_13_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 104;"
			"FilterCount = 624;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_29_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 104;"
			"FilterCount = 624;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_14_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 624;"
			"FilterCount = 104;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_28_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 624;"
			"FilterCount = 104;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_14_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 624;"
			"FilterCount = 104;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_25_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 104;"
			"FilterCount = 624;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_26_expand"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( SegLinksDepthwise2, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 32;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"expanded_conv_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 16;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_1_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 128;"
			"Width = 128;"
			"Channels = 96;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_1_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 128;"
			"Width = 128;"
			"Channels = 24;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_2_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 128;"
			"Width = 128;"
			"Channels = 144;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_2_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 64;"
			"Width = 64;"
			"Channels = 144;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_3_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 64;"
			"Width = 64;"
			"Channels = 32;"
			"FilterCount = 192;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_4_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 64;"
			"Width = 64;"
			"Channels = 192;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_4_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 192;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_6_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 64;"
			"FilterCount = 384;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_7_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 384;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_7_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 384;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_10_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 96;"
			"FilterCount = 576;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_11_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 576;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_11_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 864;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_28_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 864;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_14_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 576;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_13_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 144;"
			"FilterCount = 864;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_29_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 144;"
			"FilterCount = 864;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_14_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 864;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_25_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 144;"
			"FilterCount = 864;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_26_expand"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( PassportScanSegmentation, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_14"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 16;"
			"FilterCount = 32;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_15"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 32;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_16_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 4;"
			"DilationHeight = 4;"
			"DilationWidth = 4;"
			"Seed = 1984;",
			"conv2d_18_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 2;"
			"Seed = 1984;",
			"conv2d_17_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 8;"
			"PaddingWidth = 8;"
			"DilationHeight = 8;"
			"DilationWidth = 8;"
			"Seed = 1984;",
			"conv2d_19_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 64;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 16;"
			"PaddingWidth = 16;"
			"DilationHeight = 16;"
			"DilationWidth = 16;"
			"Seed = 1984;",
			"conv2d_20_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 4;"
			"DilationHeight = 4;"
			"DilationWidth = 4;"
			"Seed = 1984;",
			"conv2d_21_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 2;"
			"Seed = 1984;",
			"conv2d_22_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 128;"
			"FilterCount = 64;"
			"FilterHeight = 7;"
			"FilterWidth = 7;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 3;"
			"PaddingWidth = 3;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_23_"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( PassportPhotoSegmentation, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_2"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 16;"
			"FilterCount = 32;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 32;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_4_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 4;"
			"DilationHeight = 4;"
			"DilationWidth = 4;"
			"Seed = 1984;",
			"conv2d_6_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 2;"
			"Seed = 1984;",
			"conv2d_5_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 64;"
			"FilterCount = 64;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 8;"
			"PaddingWidth = 8;"
			"DilationHeight = 8;"
			"DilationWidth = 8;"
			"Seed = 1984;",
			"conv2d_7_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 96;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_8_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 128;"
			"FilterCount = 128;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_9_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 128;"
			"FilterCount = 64;"
			"FilterHeight = 7;"
			"FilterWidth = 7;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 3;"
			"PaddingWidth = 3;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_11_"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( SolitaireIDSegmentation, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_2_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_3_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_4_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 48;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 2;"
			"Seed = 1984;",
			"conv2d_5_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 48;"
			"FilterCount = 96;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 4;"
			"DilationHeight = 4;"
			"DilationWidth = 4;"
			"Seed = 1984;",
			"conv2d_6_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 96;"
			"FilterCount = 144;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 8;"
			"PaddingWidth = 8;"
			"DilationHeight = 8;"
			"DilationWidth = 8;"
			"Seed = 1984;",
			"conv2d_7_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 144;"
			"FilterCount = 192;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 16;"
			"PaddingWidth = 16;"
			"DilationHeight = 16;"
			"DilationWidth = 16;"
			"Seed = 1984;",
			"conv2d_8_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 216;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_9_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 192;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_14_"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( BarcodesPostcodes, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"separable_conv2d_2_pointwise"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"separable_conv2d_3_pointwise"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_1_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 4;"
			"DilationHeight = 4;"
			"DilationWidth = 4;"
			"Seed = 1984;",
			"conv2d_3_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 2;"
			"Seed = 1984;",
			"conv2d_2_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 8;"
			"PaddingWidth = 8;"
			"DilationHeight = 8;"
			"DilationWidth = 8;"
			"Seed = 1984;",
			"conv2d_4_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 16;"
			"PaddingWidth = 16;"
			"DilationHeight = 16;"
			"DilationWidth = 16;"
			"Seed = 1984;",
			"conv2d_5_"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( BarcodesNoPostcodes, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_11_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_12_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_13_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 4;"
			"DilationHeight = 4;"
			"DilationWidth = 4;"
			"Seed = 1984;",
			"conv2d_15_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 2;"
			"Seed = 1984;",
			"conv2d_14_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 8;"
			"PaddingWidth = 8;"
			"DilationHeight = 8;"
			"DilationWidth = 8;"
			"Seed = 1984;",
			"conv2d_16_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 16;"
			"PaddingWidth = 16;"
			"DilationHeight = 16;"
			"DilationWidth = 16;"
			"Seed = 1984;",
			"conv2d_17_"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( FindText, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 512;"
			"Width = 512;"
			"Channels = 16;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 2;"
			"StrideWidth = 2;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_9_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_10_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 24;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 2;"
			"PaddingWidth = 2;"
			"DilationHeight = 2;"
			"DilationWidth = 2;"
			"Seed = 1984;",
			"conv2d_11_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 24;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 4;"
			"PaddingWidth = 4;"
			"DilationHeight = 4;"
			"DilationWidth = 4;"
			"Seed = 1984;",
			"conv2d_12_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 8;"
			"PaddingWidth = 8;"
			"DilationHeight = 8;"
			"DilationWidth = 8;"
			"Seed = 1984;",
			"conv2d_13_"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 256;"
			"Width = 256;"
			"Channels = 16;"
			"FilterCount = 16;"
			"FilterHeight = 3;"
			"FilterWidth = 3;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 1;"
			"PaddingWidth = 1;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2d_14_"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( MobileNetV2Cifar10, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 32;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block0convShortcut"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 144;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block20conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 32;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block0conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 16;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block10convShortcut"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 16;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block10conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 384;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block31conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 96;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block10conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 24;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block11conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 32;"
			"Width = 32;"
			"Channels = 144;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block11conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 32;"
			"FilterCount = 192;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block21conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 96;"
			"FilterCount = 576;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block41conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 16;"
			"Width = 16;"
			"Channels = 192;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block21conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 320;"
			"FilterCount = 1280;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"conv2"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 192;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block30conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 64;"
			"FilterCount = 384;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block31conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 64;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block40convShortcut"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 384;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block40conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 8;"
			"Width = 8;"
			"Channels = 576;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block41conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 576;"
			"FilterCount = 160;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block50conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 160;"
			"FilterCount = 960;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block51conv1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 960;"
			"FilterCount = 160;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block51conv3"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 160;"
			"FilterCount = 320;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block6convShortcut"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 4;"
			"Width = 4;"
			"Channels = 960;"
			"FilterCount = 320;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block6conv3"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( archmobilenetv2_accurate, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 384;"
			"Width = 256;"
			"Channels = 32;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"expanded_conv_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 384;"
			"Width = 256;"
			"Channels = 16;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_1_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 192;"
			"Width = 128;"
			"Channels = 96;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_1_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 192;"
			"Width = 128;"
			"Channels = 24;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_2_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 192;"
			"Width = 128;"
			"Channels = 144;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_2_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 96;"
			"Width = 64;"
			"Channels = 144;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_3_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 96;"
			"Width = 64;"
			"Channels = 32;"
			"FilterCount = 192;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_4_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 96;"
			"Width = 64;"
			"Channels = 192;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_4_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 32;"
			"Channels = 192;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_6_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 32;"
			"Channels = 64;"
			"FilterCount = 384;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_7_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 32;"
			"Channels = 384;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_7_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 32;"
			"Channels = 384;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_10_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 16;"
			"Channels = 864;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_14_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 32;"
			"Channels = 96;"
			"FilterCount = 576;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_11_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 32;"
			"Channels = 576;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_11_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 16;"
			"Channels = 576;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_13_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 16;"
			"Channels = 144;"
			"FilterCount = 864;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_14_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 8;"
			"Channels = 864;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_25_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 8;"
			"Channels = 144;"
			"FilterCount = 864;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_26_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 6;"
			"Width = 4;"
			"Channels = 864;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_28_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 6;"
			"Width = 4;"
			"Channels = 144;"
			"FilterCount = 864;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_29_expand"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( TorchVisionMobileNetV2, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 112;"
			"Width = 112;"
			"Channels = 32;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer7"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 112;"
			"Width = 112;"
			"Channels = 16;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer9"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 56;"
			"Width = 56;"
			"Channels = 96;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer15"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 56;"
			"Width = 56;"
			"Channels = 24;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer17"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 56;"
			"Width = 56;"
			"Channels = 144;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer23"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 28;"
			"Width = 28;"
			"Channels = 144;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer32"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 28;"
			"Width = 28;"
			"Channels = 32;"
			"FilterCount = 192;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer34"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 28;"
			"Width = 28;"
			"Channels = 192;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer40"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 14;"
			"Width = 14;"
			"Channels = 192;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer58"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 14;"
			"Width = 14;"
			"Channels = 64;"
			"FilterCount = 384;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer60"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 14;"
			"Width = 14;"
			"Channels = 384;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer66"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 14;"
			"Width = 14;"
			"Channels = 384;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer93"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 14;"
			"Width = 14;"
			"Channels = 96;"
			"FilterCount = 576;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer95"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 14;"
			"Width = 14;"
			"Channels = 576;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer101"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 7;"
			"Width = 7;"
			"Channels = 576;"
			"FilterCount = 160;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer119"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 7;"
			"Width = 7;"
			"Channels = 320;"
			"FilterCount = 1280;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer147"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 7;"
			"Width = 7;"
			"Channels = 160;"
			"FilterCount = 960;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer121"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 7;"
			"Width = 7;"
			"Channels = 960;"
			"FilterCount = 160;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer127"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 7;"
			"Width = 7;"
			"Channels = 960;"
			"FilterCount = 320;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"FineAILayer145"
		)
	), BlockedConvTestNameGenerator() );

INSTANTIATE_TEST_SUITE_P( ScreenshotFeatureExtractor, CBlockedConvTest,
	::testing::Values(
		CTestParams(
			"Batch = 1;"
			"Height = 192;"
			"Width = 192;"
			"Channels = 32;"
			"FilterCount = 16;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"expanded_conv_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 192;"
			"Width = 192;"
			"Channels = 16;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_1_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 96;"
			"Width = 96;"
			"Channels = 96;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_1_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 96;"
			"Width = 96;"
			"Channels = 24;"
			"FilterCount = 144;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_2_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 96;"
			"Width = 96;"
			"Channels = 144;"
			"FilterCount = 24;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_2_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 48;"
			"Channels = 144;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_3_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 48;"
			"Channels = 32;"
			"FilterCount = 192;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_4_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 48;"
			"Width = 48;"
			"Channels = 192;"
			"FilterCount = 32;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_4_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 24;"
			"Channels = 192;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_6_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 24;"
			"Channels = 64;"
			"FilterCount = 384;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_7_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 24;"
			"Channels = 384;"
			"FilterCount = 64;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_7_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 12;"
			"Channels = 320;"
			"FilterCount = 1280;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"Conv_1"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 24;"
			"Channels = 384;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_10_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 24;"
			"Channels = 96;"
			"FilterCount = 576;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_11_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 24;"
			"Width = 24;"
			"Channels = 576;"
			"FilterCount = 96;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_11_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 12;"
			"Channels = 576;"
			"FilterCount = 160;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_13_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 12;"
			"Channels = 160;"
			"FilterCount = 960;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_14_expand"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 12;"
			"Channels = 960;"
			"FilterCount = 160;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_14_project"
		),
		CTestParams(
			"Batch = 1;"
			"Height = 12;"
			"Width = 12;"
			"Channels = 960;"
			"FilterCount = 320;"
			"FilterHeight = 1;"
			"FilterWidth = 1;"
			"StrideHeight = 1;"
			"StrideWidth = 1;"
			"PaddingHeight = 0;"
			"PaddingWidth = 0;"
			"DilationHeight = 1;"
			"DilationWidth = 1;"
			"Seed = 1984;",
			"block_16_project"
		)
	), BlockedConvTestNameGenerator() );
