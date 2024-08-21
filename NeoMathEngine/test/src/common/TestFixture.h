/* Copyright © 2017-2024 ABBYY

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
#include <NeoMathEngine/NeoMathEngine.h>

#include <cmath>
#include <algorithm>
#include <cassert>

using namespace NeoML;

namespace NeoMLTest {

int RunTests( int argc, char* argv[] );

IMathEngine& MathEngine();

//------------------------------------------------------------------------------------------------------------

// Yellow output
class NeoMLTestHighlightedOutput final {
public:
	NeoMLTestHighlightedOutput( ::std::ostream& _log ) : log( _log ) { log << "\u001b[33m"; }
	~NeoMLTestHighlightedOutput() { log << "\u001b[0m"; }

	template <typename T> ::std::ostream& operator<<( T t ) { return log << t; }
private:
	::std::ostream& log;
};

#define NEOML_HILIGHT( log )   NeoMLTestHighlightedOutput( log )

inline ::std::ostream& operator<<( ::std::ostream& s, TMathEngineType met )
{
	switch( met ) {
		case MET_Cpu: s << "MET_Cpu"; break;
		case MET_Cuda: s << "MET_Cuda"; break;
		case MET_Metal: s << "MET_Metal"; break;
		case MET_Vulkan: s << "MET_Vulkan"; break;
		default: ASSERT_EXPR( false );
	}
	return s;
}

//------------------------------------------------------------------------------------------------------------

inline bool FloatEq( float val1, float val2, float precision = 1e-05 )
{
	if( val1 >= FLT_MAX ) {
		return val2 >= FLT_MAX;
	}
	if( val1 <= -FLT_MAX ) {
		return val2 <= -FLT_MAX;
	}
	if( std::isnan( val1 ) ) {
		return std::isnan( val2 ) != 0;
	}
	if( abs( val2 ) < precision && abs( val1 ) < precision ) {
		return true;
	}
	return abs( val1 - val2 ) < precision || abs( ( val1 - val2 ) / ( val2 == 0 ? FLT_EPSILON : val2 ) ) < precision;
}

//------------------------------------------------------------------------------------------------------------

#define FLT_MIN_LOG -87.33654474f
#define FLT_MAX_LOG 88.f

inline float ExponentFunc(float f)
{
	if (f < FLT_MIN_LOG) {
		return 0;
	}
	else if (f > FLT_MAX_LOG) {
		return FLT_MAX;
	}
	else {
		return expf(f);
	}
}

//------------------------------------------------------------------------------------------------------------

#define CARRAY_WRAPPER(TYPE, arr) CBufferWrapper<TYPE>( MathEngine(), ( arr.data() ), ( static_cast<int>( arr.size() ) ) )
#define CARRAY_FLOAT_WRAPPER(arr) CARRAY_WRAPPER(float, arr)
#define CARRAY_INT_WRAPPER(arr) CARRAY_WRAPPER(int, arr)

#define ARR_WRAPPER(TYPE, arr) CBufferWrapper<TYPE>( MathEngine(), (arr), (int)sizeof(arr) / sizeof(TYPE) )
#define FLOAT_WRAPPER(arr) CFloatWrapper( MathEngine(), (arr), (int)sizeof(arr) / sizeof(float) )
#define FLOAT_WRAPPER_MATHENGINE(mathEngine, arr) CFloatWrapper( mathEngine, (arr), (int)sizeof(arr) / sizeof(float) )
#define INT_WRAPPER(arr) CIntWrapper( MathEngine(), (arr), (int)sizeof(arr) / sizeof(int) )
#define INT_WRAPPER_MATHENGINE(mathEngine, arr) CIntWrapper( mathEngine, (arr), (int)sizeof(arr) / sizeof(int) )

#define CREATE_FILL_ARRAY( TYPE, arr, min, max, size, random ) \
	std::vector<TYPE> arr; \
	arr.resize( size ); \
	for( auto i = decltype( size ){0}; i < size; ++i ) { \
		arr[i] = static_cast<TYPE>( random.Uniform( min, max ) ); \
	}

#define CREATE_FILL_FLOAT_ARRAY( arr, min, max, size, random ) \
	CREATE_FILL_ARRAY( float, arr, min, max, size, random )

// Leaving CREATE_FILL_INT_ARRAY as it was before CREATE_FILL_ARRAY
// for the sake of backward compatibility
#define CREATE_FILL_INT_ARRAY( arr, min, max, size, random ) \
	std::vector<int> arr; \
	arr.resize( size ); \
	for( auto i = decltype( size ){0}; i < size; ++i ) { \
		arr[i] = random.UniformInt( min, max ); \
	}

//------------------------------------------------------------------------------------------------------------

template<class T>
class CBufferWrapper final {
public:
	CBufferWrapper( IMathEngine& _mathEngine, T* _data, int _size ) :
		mathEngine( _mathEngine ),
		isCopyBack( false ),
		size( _size ),
		data( _data ),
		mathData( mathEngine.HeapAlloc( size * sizeof( T ) ) )
	{
		mathEngine.DataExchangeTyped<T>( CTypedMemoryHandle<T>( mathData ), data, size );
	}
	~CBufferWrapper()
	{
		if( isCopyBack ) {
			mathEngine.DataExchangeTyped<T>( data, CTypedMemoryHandle<T>( mathData ), size );
		}
		mathEngine.HeapFree( mathData );
	}

	operator CTypedMemoryHandle<T>() const { isCopyBack = true; return CTypedMemoryHandle<T>( mathData ); }
	operator CTypedMemoryHandle<const T>() const { return CTypedMemoryHandle<const T>( mathData ); }

private:
	IMathEngine& mathEngine;
	mutable bool isCopyBack;
	int size;
	T* data;
	CTypedMemoryHandle<T> mathData;
};

typedef CBufferWrapper<float> CFloatWrapper;
typedef CBufferWrapper<int> CIntWrapper;

//------------------------------------------------------------------------------------------------------------

template <class T>
class CBlob final {
public:
	CBlob( IMathEngine& mathEngine, int batchLength, int batchWidth, int listSize, int height, int width, int depth, int channels );
	CBlob( IMathEngine& mathEngine, int objectCount, int height, int width, int depth, int channelsCount ) :
		CBlob( mathEngine, 1, objectCount, 1, height, width, depth, channelsCount )
	{}
	CBlob( IMathEngine& mathEngine, int objectCount, int height, int width, int channelsCount ) :
		CBlob( mathEngine, 1, objectCount, 1, height, width, 1, channelsCount )
	{}

	const CBlobDesc& GetDesc() const { return desc; }
	CTypedMemoryHandle<T> GetData() const { return data.GetHandle(); }
	int GetDataSize() const { return data.Size(); }

	void CopyFrom( const T* src ) { data.GetMathEngine()->DataExchangeRaw( GetData(), src, data.Size() * sizeof( T ) ); }
	void CopyTo( T* dst ) const { data.GetMathEngine()->DataExchangeRaw( dst, GetData(), data.Size() * sizeof( T ) ); }

private:
	CBlobDesc desc;
	CMemoryHandleVar<T> data;
};

template<class T>
inline CBlob<T>::CBlob( IMathEngine& mathEngine,
	int batchLength, int batchWidth, int listSize, int height, int width, int depth, int channels ) :
	data( mathEngine, batchLength * batchWidth * listSize * height * width * depth * channels )
{
	switch( CBlobType<T>::GetType() ) {
		case CT_Float:
			desc.SetDataType( CT_Float );
			break;
		case CT_Int:
			desc.SetDataType( CT_Int );
			break;
		default:
			ASSERT_EXPR( false );
	}
	desc.SetDimSize( BD_BatchLength, batchLength );
	desc.SetDimSize( BD_BatchWidth, batchWidth );
	desc.SetDimSize( BD_ListSize, listSize );
	desc.SetDimSize( BD_Height, height );
	desc.SetDimSize( BD_Width, width );
	desc.SetDimSize( BD_Depth, depth );
	desc.SetDimSize( BD_Channels, channels );
}

typedef CBlob<float> CFloatBlob;
typedef CBlob<int> CIntBlob;

inline int GetFlatIndex( const CBlobDesc& blob, int seq,
	int batch, int list, int channel, int depth, int row, int column )
{
	return ( list + blob.ListSize() * ( batch + blob.BatchWidth() * seq ) ) * blob.ObjectSize()
		+ channel + blob.Channels() * ( depth + blob.Depth() * ( column + row * blob.Width() ) );
}

//------------------------------------------------------------------------------------------------------------

class CTestFixture : public ::testing::Test {
};

class CTestFixtureWithParams : public CTestFixture, public ::testing::WithParamInterface<CTestParams> {
};

//------------------------------------------------------------------------------------------------------------

#define RUN_TEST_IMPL( impl ) { \
	const CTestParams& params = GetParam(); \
	const int testCount = params.GetValue<int>( "TestCount" ); \
	for( int test = 0; test < testCount; ++test ) { \
		impl ( params, 282 + test * 10000 + test % 3  ); \
	} } \

//------------------------------------------------------------------------------------------------------------

// The class to generate random values
// It uses the complementary-multiply-with-carry algorithm
// C lag-1024, multiplier(a) = 108798, initial carry(c) = 12345678
class CRandom final {
public:
	explicit CRandom( unsigned int seed = 0xBADF00D ) { srand( seed ); }

	// Returns the next random value
	unsigned int Next() { return ( rand() << 16 ) + rand(); }

	// Returns a double value from a uniform distribution in [ min, max ) range
	// If min == max, min is returned
	double Uniform( double min, double max ) { return min + ( max - min ) * Next() / 4294967296.; }

	// Returns an int value from a uniform distribution in [ min, max ] range. Note that max return value is possible!
	// If min == max, min is returned
	int UniformInt( int min, int max ) { return (int)( min + (unsigned int)( ( ( static_cast<unsigned long long>( max ) - min + 1 ) * Next() ) >> 32 ) ); }
};

//------------------------------------------------------------------------------------------------------------

class CSparseMatrix final {
public:
	CSparseMatrix( IMathEngine& mathEngine, const std::vector<int>& _rows,
			const std::vector<int>& _columns, const std::vector<float>& _values ) :
		elementCount( static_cast<int>( _values.size() ) ),
		rows( mathEngine, _rows.size() ),
		columns( mathEngine, _columns.size() ),
		values( mathEngine, _values.size() )
	{
		mathEngine.DataExchangeTyped<int>( rows, _rows.data(), _rows.size() );
		mathEngine.DataExchangeTyped<int>( columns, _columns.data(), _columns.size() );
		mathEngine.DataExchangeTyped<float>( values, _values.data(), _values.size() );
	}

	CSparseMatrixDesc Desc() const
	{
		CSparseMatrixDesc desc;
		desc.ElementCount = elementCount;
		desc.Rows = rows;
		desc.Columns = columns;
		desc.Values = values;
		return desc;
	}

private:
	int elementCount;
	CIntHandleVar rows;
	CIntHandleVar columns;
	CFloatHandleVar values;
};

} // namespace NeoMLTest
