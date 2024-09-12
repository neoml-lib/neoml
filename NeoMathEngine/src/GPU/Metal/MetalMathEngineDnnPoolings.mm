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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_METAL

#include <MetalMathEngine.h>
#include <MathEngineDnnPoolings.h>
#include <MathEngineCommon.h>
#include <MetalKernel.h>

@import Foundation;
@import MetalKit;

namespace NeoML {

//--------------------------------------------------------------------------------------------------------------------------
// Max pooling

CMaxPoolingDesc* CMetalMathEngine::InitMaxPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	const CBlobDesc& result )
{
	CCommonMaxPoolingDesc* desc = new CCommonMaxPoolingDesc( source, result, filterHeight, filterWidth, strideHeight, strideWidth );
	return desc;
}

void CMetalMathEngine::BlobMaxPooling( const CMaxPoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonMaxPoolingDesc& desc = static_cast<const CCommonMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int totalChannels = result.Depth() * result.Channels();

    int height = result.Height();
    int width = result.Width() * totalChannels;

    C3DKernel kernel( *queue, ( maxIndicesData == 0 ) ? "cubeKernelBlobMaxPoolingNoIndices" : "cubeKernelBlobMaxPooling", 1, 1, 1, result.ObjectCount(), height, width );
    kernel.SetParam( source, 0 );
    kernel.SetParam( sourceData, 1 );
    kernel.SetParam( desc.FilterHeight, 2 );
    kernel.SetParam( desc.FilterWidth, 3 );
    kernel.SetParam( desc.StrideHeight, 4 );
    kernel.SetParam( desc.StrideWidth, 5 );
    kernel.SetParam( result, 6 );
    kernel.SetParam( resultData, 7 );
    if( maxIndicesData != 0 ) {
        kernel.SetParam( *maxIndicesData, 8 );
    }
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::BlobMaxPoolingBackward( const CMaxPoolingDesc&,
	const CConstFloatHandle&, const CConstIntHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//--------------------------------------------------------------------------------------------------------------------------
// Mean pooling

CMeanPoolingDesc* CMetalMathEngine::InitMeanPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	const CBlobDesc& result )
{
	CCommonMeanPoolingDesc* desc = new CCommonMeanPoolingDesc( source, result, filterHeight, filterWidth, strideHeight, strideWidth );
	return desc;
}

void CMetalMathEngine::BlobMeanPooling( const CMeanPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonMeanPoolingDesc& desc = static_cast<const CCommonMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;
	const int totalChannels = result.Depth() * result.Channels();
     
    int height = result.Height();
    int width = result.Width() * totalChannels;

    C3DKernel kernel( *queue, "cubeKernelBlobMeanPooling", 1, 1, 1, result.ObjectCount(), height, width );
    kernel.SetParam( source, 0 );
    kernel.SetParam( sourceData, 1 );
    kernel.SetParam( desc.FilterHeight, 2 );
    kernel.SetParam( desc.FilterWidth, 3 );
    kernel.SetParam( desc.StrideHeight, 4 );
    kernel.SetParam( desc.StrideWidth, 5 );
    kernel.SetParam( result, 6 );
    kernel.SetParam( resultData, 7 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::BlobMeanPoolingBackward( const CMeanPoolingDesc&, const CConstFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//--------------------------------------------------------------------------------------------------------------------------
// Global max-over-time pooling

CGlobalMaxOverTimePoolingDesc* CMetalMathEngine::InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result )
{
	CCommonGlobalMaxOverTimePoolingDesc* desc = new CCommonGlobalMaxOverTimePoolingDesc( source, result );
	return desc;
}

void CMetalMathEngine::BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonGlobalMaxOverTimePoolingDesc& desc = static_cast<const CCommonGlobalMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

    const int objectsCount = source.BatchLength();
    const int objectSize = source.BlobSize() / objectsCount;
    
    C1DKernel kernel( *queue, ( maxIndicesData == 0 ) ? "vectorKernelBlobGlobalMaxOverTimePooling" : "vectorKernelBlobGlobalMaxOverTimePoolingWithIndex", 1, objectSize );
    kernel.SetParam( source, 0 );
    kernel.SetParam( sourceData, 1 );
    kernel.SetParam( result, 2 );
    kernel.SetParam( resultData, 3 );
    if( maxIndicesData != 0 ) {
        kernel.SetParam( *maxIndicesData, 4 );
    }
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc&,
	const CConstFloatHandle&, const CConstIntHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//--------------------------------------------------------------------------------------------------------------------------
// Global-max pooling

CGlobalMaxPoolingDesc* CMetalMathEngine::InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result )
{
	ASSERT_EXPR( result.ObjectCount() == source.ObjectCount() && maxIndices.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( maxIndices.ObjectSize() == result.ObjectSize() );

	CCommonGlobalMaxPoolingDesc* desc = new CCommonGlobalMaxPoolingDesc( source, result, maxIndices );
	return desc;
}

void CMetalMathEngine::BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CIntHandle& maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonGlobalMaxPoolingDesc& desc = static_cast<const CCommonGlobalMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
    const CBlobDesc& maxIndices = desc.MaxIndices;
	const CBlobDesc& result = desc.Result;

    const int poolSize = source.Depth() * source.Height() * source.Width();
	const int maxCount = result.Depth() * result.Height() * result.Width();
	const int sharedMemoryPerThread = 4 * maxCount * sizeof(float);

    C2DKernel kernel( *queue, "matrixKernelBlobGlobalMaxPooling", 1, 1, source.ObjectCount() * source.Channels(), poolSize );
    kernel.SetParam( source, 0 );
    kernel.SetParam( sourceData, 1 );
    kernel.SetParam( maxIndices, 2 );
    kernel.SetParam( maxIndicesData, 3 );
    kernel.SetParam( result, 4 );
    kernel.SetParam( resultData, 5 );
    kernel.SetParam( poolSize, 6 );
    kernel.SetParam( maxCount, 7 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sharedMemoryPerThread, 8 );
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc&,
	const CConstFloatHandle&, const CConstIntHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//--------------------------------------------------------------------------------------------------------------------------
// 3D-max pooling

C3dMaxPoolingDesc* CMetalMathEngine::Init3dMaxPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& result )
{
	CCommon3dMaxPoolingDesc* desc = new CCommon3dMaxPoolingDesc( source, result, filterHeight, filterWidth, filterDepth,
		strideHeight, strideWidth, strideDepth );
	return desc;
}

void CMetalMathEngine::Blob3dMaxPooling( const C3dMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 || maxIndicesData->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommon3dMaxPoolingDesc& desc = static_cast<const CCommon3dMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

    C3DKernel kernel( *queue, ( maxIndicesData == 0 ) ? "cubeKernelBlob3dMaxPoolingNoIndices" : "cubeKernelBlob3dMaxPooling", 1, 1, 1,
        result.ObjectCount(), result.Channels(), result.Depth() * result.Height() * result.Width() );
    kernel.SetParam( source, 0 );
    kernel.SetParam( sourceData, 1 );
    kernel.SetParam( desc.FilterHeight, 2 );
    kernel.SetParam( desc.FilterWidth, 3 );
    kernel.SetParam( desc.FilterDepth, 4 );
    kernel.SetParam( desc.StrideHeight, 5 );
    kernel.SetParam( desc.StrideWidth, 6 );
    kernel.SetParam( desc.StrideDepth, 7 );
    kernel.SetParam( result, 8 );
    kernel.SetParam( resultData, 9 );
    if( maxIndicesData != 0 ) {
        kernel.SetParam( *maxIndicesData, 10 );
    }
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc&,
	const CConstFloatHandle&, const CConstIntHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//--------------------------------------------------------------------------------------------------------------------------
// 3D-mean pooling

C3dMeanPoolingDesc* CMetalMathEngine::Init3dMeanPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& result )
{
	CCommon3dMeanPoolingDesc* desc = new CCommon3dMeanPoolingDesc( source, result, filterHeight, filterWidth, filterDepth,
		strideHeight, strideWidth, strideDepth );
	return desc;
}

void CMetalMathEngine::Blob3dMeanPooling( const C3dMeanPoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommon3dMeanPoolingDesc& desc = static_cast<const CCommon3dMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

    C3DKernel kernel( *queue, "cubeKernelBlob3dMeanPooling",
        1, 1, 1, result.ObjectCount(), result.Channels(), result.Depth() * result.Height() * result.Width() );
    kernel.SetParam( source, 0 );
    kernel.SetParam( sourceData, 1 );
    kernel.SetParam( desc.FilterHeight, 2 );
    kernel.SetParam( desc.FilterWidth, 3 );
    kernel.SetParam( desc.FilterDepth, 4 );
    kernel.SetParam( desc.StrideHeight, 5 );
    kernel.SetParam( desc.StrideWidth, 6 );
    kernel.SetParam( desc.StrideDepth, 7 );
    kernel.SetParam( result, 8 );
    kernel.SetParam( resultData, 9 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc&,
	const CConstFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//--------------------------------------------------------------------------------------------------------------------------
// Max-Over-Time Pooling

CMaxOverTimePoolingDesc* CMetalMathEngine::InitMaxOverTimePooling( const CBlobDesc& source,
	int filterLen, int strideLen, const CBlobDesc& result )
{
	int outLen = ( source.BatchLength() - filterLen ) / strideLen + 1;
	ASSERT_EXPR( result.BatchLength() == outLen );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ObjectSize() == source.ObjectSize() );

	CCommonMaxOverTimePoolingDesc* desc = new CCommonMaxOverTimePoolingDesc( source, result, filterLen, strideLen );
	return desc;
}

void CMetalMathEngine::BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( maxIndicesData == 0 );
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonMaxOverTimePoolingDesc& desc = static_cast<const CCommonMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

    C2DKernel kernel( *queue, (maxIndicesData != 0) ? "matrixKernelBlobMaxOverTimePooling" : "matrixKernelBlobMaxOverTimePoolingNoIndexes",
        1, 1, result.BlobSize(), desc.FilterLen );
    kernel.SetParam( source, 0 );
    kernel.SetParam( sourceData, 1 );
    kernel.SetParam( desc.FilterLen, 2 );
    kernel.SetParam( desc.StrideLen, 3 );
    kernel.SetParam( result, 4 );
    kernel.SetParam( resultData, 5 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 6 );
    if( maxIndicesData != 0 ) {
        kernel.SetParam( *maxIndicesData, 7 );
        kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(int), 8 );
    }
    
    // threadgroupCount.width = 1;
    ASSERT_EXPR( kernel.Run( 0, 0, 1 ) );
}

void CMetalMathEngine::BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc&,
	const CConstFloatHandle&, const CConstIntHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_METAL
