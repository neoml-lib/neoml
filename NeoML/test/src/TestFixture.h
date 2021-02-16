/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <cmath>

namespace NeoMLTest {

CString GetTestDataFilePath( const CString& relativePath, const CString& fileName );

// Get global MathEngine.
IMathEngine& MathEngine();
// Delete global MathEngine.
void DeleteMathEngine();
// Get global MathEngine type.
TMathEngineType MathEngineType();

void SetPlatformEnv( void* platformEnv );

void* GetPlatformEnv();

NeoML::IMathEngine* CreateMathEngine( TMathEngineType type, std::size_t memoryLimit = 0u, int threadCount = 0 );

#ifdef NEOML_USE_FINEOBJ
int RunTests( int argc, wchar_t* argv[], void* platformEnv = nullptr );
#else
int RunTests( int argc, char* argv[], void* platformEnv = nullptr );
#endif

//------------------------------------------------------------------------------------------------------------

#define FLT_MIN_LOG -87.33654474f
#define FLT_MAX_LOG 88.f

#define CARRAY_FLOAT_WRAPPER(arr) CFloatWrapper( MathEngine(), (arr.GetPtr()), arr.Size() )
#define CARRAY_INT_WRAPPER(arr) CIntWrapper( MathEngine(), (arr.GetPtr()), arr.Size() )

#define FLOAT_WRAPPER(arr) CFloatWrapper( MathEngine(), (arr), (int)sizeof(arr) / sizeof(float) )
#define FLOAT_WRAPPER_MATHENGINE(mathEngine, arr) CFloatWrapper( mathEngine, (arr), (int)sizeof(arr) / sizeof(float) )
#define INT_WRAPPER(arr) CIntWrapper( MathEngine(), (arr), (int)sizeof(arr) / sizeof(int) )
#define INT_WRAPPER_MATHENGINE(mathEngine, arr) CIntWrapper( mathEngine, (arr), (int)sizeof(arr) / sizeof(int) )

#define CREATE_FILL_FLOAT_ARRAY(arr, min, max, size, random) \
	CArray<float> arr; \
	arr.SetSize( size ); \
	for(int i = 0; i < size; ++i) { \
        arr[i] = static_cast<float>( random.Uniform( min, max ) ); \
	}

#define CREATE_FILL_INT_ARRAY(arr, min, max, size, random) \
	CArray<int> arr; \
	arr.SetSize( size ); \
	for(int i = 0; i < size; ++i) { \
        arr[i] = random.UniformInt( min, max ); \
	}

//------------------------------------------------------------------------------------------------------------

// класс для передачи простых float масивов
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

//------------------------------------------------------------------------------------------------------------

inline int GetFlatIndex( const CDnnBlob& blob, int seq, int batch, int list, int channel, int depth,
	int row, int column )
{
	return ( list + blob.GetListSize() * ( batch + blob.GetBatchWidth() * seq ) ) * blob.GetObjectSize()
		+ channel + blob.GetChannelsCount() * ( depth + blob.GetDepth() * ( column + row * blob.GetWidth() ) );
}

inline bool FloatEqImpl( float val1, float val2, float precision = 1e-05 )
{
	if(val1 >= FLT_MAX) {
		return val2 >= FLT_MAX;
	}

	if(val1 <= -FLT_MAX) {
		return val2 <= -FLT_MAX;
	}

	if(std::isnan(val1)) {
		return std::isnan(val2) != 0;
	}

	if(abs(val2) < precision && abs(val1) < precision) {
		return true;
	}

	return abs(val1 - val2) < precision || abs((val1 - val2) / (val2 == 0 ? FLT_EPSILON : val2)) < precision;
}

inline bool FloatEq( float val1, float val2, float precision = 1e-05 )
{
	bool ret = FloatEqImpl(val1, val2, precision);
	NeoPresume(ret);
	return ret;
}

//------------------------------------------------------------------------------------------------------------

// Создает слой с данным именем и добавляет его к сети.
template<class T>
CPtr<T> AddLayer( const CPtr<T>& layer, const CString& layerName, CDnn& net )
{
	layer->SetName( layerName );
	net.AddLayer( *layer );
	return layer;
}

template<class T>
CPtr<T> AddLayer( const CString& layerName, CDnn& net )
{
	CPtr<T> layer = new T( net.GetMathEngine() );
	return AddLayer( layer, layerName, net );
}

template<class T>
CPtr<T> AddLayer( const CPtr<T>& layer, const CString& layerName, const CArray<CBaseLayer*>& input )
{
	NeoAssert( !input.IsEmpty() );
	AddLayer( layer, layerName, *input[0]->GetDnn() );
	for( int i = 0; i < input.Size(); ++i ) {
		layer->Connect( i, *input[i] );
	}
	return layer;
}

template<class T>
CPtr<T> AddLayer( const CString& layerName, const CArray<CBaseLayer*>& input )
{
	NeoAssert( !input.IsEmpty() );
	CPtr<T> layer = new T( input[0]->GetDnn()->GetMathEngine() );
	return AddLayer( layer, layerName, input );
}

//------------------------------------------------------------------------------------------------------------

class CNeoMLTestFixture : public ::testing::Test {
};

} // namespace NeoMLTest
