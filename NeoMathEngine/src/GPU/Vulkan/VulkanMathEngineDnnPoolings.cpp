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

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_VULKAN

#include <VulkanMathEngine.h>
#include <MathEngineCommon.h>
#include <VulkanShader.h>
#include <MathEngineDnnPoolings.h>

#include <climits>
#include <cfloat>

namespace NeoML {

// Include the shader code
#include <shaders/generated/BlobMaxPooling.h>
#include <shaders/generated/Blob3dMaxPoolingNoIndices.h>
#include <shaders/generated/Blob3dMeanPooling.h>
#include <shaders/generated/BlobMeanPooling.h>
#include <shaders/generated/BlobMaxOverTimePoolingNoIndices.h>
#include <shaders/generated/BlobGlobalMaxPooling.h>

//------------------------------------------------------------------------------------------------------------
// max pooling

CMaxPoolingDesc* CVulkanMathEngine::InitMaxPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	const CBlobDesc& result )
{
	CCommonMaxPoolingDesc* desc = new CCommonMaxPoolingDesc( source, result, filterHeight, filterWidth, strideHeight, strideWidth );
	return desc;
}

void CVulkanMathEngine::BlobMaxPooling( const CMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonMaxPoolingDesc& desc = static_cast<const CCommonMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	CMemoryHandle bufs[2] = { sourceData, resultData };
	size_t sizes[2] = { source.BlobSize() * sizeof(float), result.BlobSize() * sizeof(float) };

	PARAM_STRUCT(BlobMaxPooling) param = { { desc.StrideWidth, desc.StrideHeight }, { desc.FilterWidth, desc.FilterHeight },
		result.ObjectCount(), result.Channels() * result.Depth(), result.Height(), result.Width(),
		source.Height(), source.Width() };

	runShader(shaderLoader->GET_SHADER_DATA(BlobMaxPooling, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2,
		result.Width(), result.ObjectCount() * result.Height(), result.Channels() * result.Depth());
}

void CVulkanMathEngine::BlobMaxPoolingBackward( const CMaxPoolingDesc&, const CFloatHandle&, const CIntHandle&,
	const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// 3d max pooling

C3dMaxPoolingDesc* CVulkanMathEngine::Init3dMaxPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& result )
{
	CCommon3dMaxPoolingDesc* desc = new CCommon3dMaxPoolingDesc( source, result, filterHeight, filterWidth, filterDepth,
		strideHeight, strideWidth, strideDepth );
	return desc;
}

void CVulkanMathEngine::Blob3dMaxPooling( const C3dMaxPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == nullptr );

	const CCommon3dMaxPoolingDesc& desc = static_cast<const CCommon3dMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	CMemoryHandle bufs[2] = { sourceData, resultData };
	size_t sizes[2] = { source.BlobSize() * sizeof(float), source.BlobSize() * sizeof(float) };

	PARAM_STRUCT(Blob3dMaxPoolingNoIndices) param = { 
		desc.StrideHeight, desc.StrideWidth, desc.StrideDepth,
		desc.FilterHeight, desc.FilterWidth, desc.FilterDepth,
		source.Height(), source.Width(), source.Depth(),
		result.Height(), result.Width(), result.Depth(), 
		result.Channels(), result.ObjectCount()
	};

	runShader(shaderLoader->GET_SHADER_DATA(Blob3dMaxPoolingNoIndices, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2,
		result.Width() * result.Height() * result.Depth(), result.Channels(), result.ObjectCount());
}

void CVulkanMathEngine::Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc&, const CFloatHandle&, const CIntHandle&,
	const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// 3d mean pooling

C3dMeanPoolingDesc* CVulkanMathEngine::Init3dMeanPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& result )
{
	CCommon3dMeanPoolingDesc* desc = new CCommon3dMeanPoolingDesc( source, result, filterHeight, filterWidth, filterDepth,
		strideHeight, strideWidth, strideDepth );
	return desc;
}

void CVulkanMathEngine::Blob3dMeanPooling( const C3dMeanPoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommon3dMeanPoolingDesc& desc = static_cast<const CCommon3dMeanPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	CMemoryHandle bufs[2] = { sourceData, resultData };
	size_t sizes[2] = { source.BlobSize() * sizeof(float), source.BlobSize() * sizeof(float) };

	PARAM_STRUCT(Blob3dMeanPooling) param = { 
		desc.StrideHeight, desc.StrideWidth, desc.StrideDepth,
		desc.FilterHeight, desc.FilterWidth, desc.FilterDepth,
		source.Height(), source.Width(), source.Depth(),
		result.Height(), result.Width(), result.Depth(), 
		result.Channels(), result.ObjectCount()
	};

	runShader( shaderLoader->GET_SHADER_DATA(Blob3dMeanPooling, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2,
		result.Width() * result.Height() * result.Depth(), result.Channels(), result.ObjectCount() );
}

void CVulkanMathEngine::Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc&, const CFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// Max over time pooling

CMaxOverTimePoolingDesc* CVulkanMathEngine::InitMaxOverTimePooling( const CBlobDesc& source,
	int filterLen, int strideLen, const CBlobDesc& result )
{
	int outLen = ( source.BatchLength() - filterLen ) / strideLen + 1;
	ASSERT_EXPR( result.BatchLength() == outLen );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.ObjectSize() == source.ObjectSize() );

	CCommonMaxOverTimePoolingDesc* desc = new CCommonMaxOverTimePoolingDesc( source, result, filterLen, strideLen );
	return desc;
}

void CVulkanMathEngine::BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( maxIndicesData == 0 );
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 );

	const CCommonMaxOverTimePoolingDesc& desc = static_cast<const CCommonMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	CMemoryHandle bufs[2] = { sourceData, resultData };
	size_t sizes[2] = { source.BlobSize() * sizeof(float), source.BlobSize() * sizeof(float) };

	PARAM_STRUCT(BlobMaxOverTimePoolingNoIndices) param = { 
		result.BlobSize(), result.BatchWidth(), result.ObjectSize(), desc.FilterLen, desc.StrideLen
	};

	runShader( shaderLoader->GET_SHADER_DATA(BlobMaxOverTimePoolingNoIndices, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2,
		result.BlobSize(), 1, 1 );
}

void CVulkanMathEngine::BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc&, const CFloatHandle&,
	const CIntHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// Global max pooling

CGlobalMaxPoolingDesc* CVulkanMathEngine::InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result )
{
	ASSERT_EXPR( result.ObjectCount() == source.ObjectCount() && maxIndices.ObjectCount() == result.ObjectCount() );
	ASSERT_EXPR( maxIndices.ObjectSize() == result.ObjectSize() );

	CCommonGlobalMaxPoolingDesc* desc = new CCommonGlobalMaxPoolingDesc( source, result, maxIndices );
	return desc;
}

void CVulkanMathEngine::BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CIntHandle& maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonGlobalMaxPoolingDesc& desc = static_cast<const CCommonGlobalMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	VectorFill(resultData, -FLT_MAX, result.BlobSize());
	VectorFill(maxIndicesData, -1, result.BlobSize());

	CMemoryHandle bufs[3] = { sourceData, maxIndicesData, resultData };
	size_t sizes[3] = { source.BlobSize() * sizeof(float), result.BlobSize() * sizeof(int), result.BlobSize() * sizeof(float) };

	int poolSize = source.Height() * source.Width() * source.Depth();
	int maxCount = result.Height() * result.Width() * result.Depth();

	PARAM_STRUCT(BlobGlobalMaxPooling) param =
		{ maxCount, source.ObjectCount(), source.Channels(), poolSize * source.Channels(), maxCount * result.Channels(), poolSize };

	runShader( shaderLoader->GET_SHADER_DATA(BlobGlobalMaxPooling, true, 0, 0, 3),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 3, result.ObjectCount(), result.Channels(), 1 );
}

void CVulkanMathEngine::BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc&, const CFloatHandle&, const CIntHandle&,
	const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// mean pooling

CMeanPoolingDesc* CVulkanMathEngine::InitMeanPooling( const CBlobDesc& source,
	int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	const CBlobDesc& result )
{
	CCommonMeanPoolingDesc* desc = new CCommonMeanPoolingDesc( source, result, filterHeight, filterWidth, strideHeight, strideWidth );
	return desc;
}

void CVulkanMathEngine::BlobMeanPooling( const CMeanPoolingDesc& poolingDesc, const CFloatHandle& sourceData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	CCommonMeanPoolingDesc desc = static_cast<const CCommonMeanPoolingDesc &>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	CMemoryHandle bufs[2] = { sourceData, resultData };
	size_t sizes[2] = { source.BlobSize() * sizeof(float), result.BlobSize() * sizeof(float) };

	PARAM_STRUCT(BlobMeanPooling) param = { { desc.StrideWidth, desc.StrideHeight}, { desc.FilterWidth, desc.FilterHeight },
		result.ObjectCount(), result.Channels() * result.Depth(), result.Height(), result.Width(),
		source.Height(), source.Width() };

	runShader( shaderLoader->GET_SHADER_DATA(BlobMeanPooling, true, 0, 0, 2),
		&param, sizeof(param), 0, 0, 0, 0, bufs, sizes, 2,
		result.Width(), result.ObjectCount() * result.Height(), result.Channels() * result.Depth() );
}

void CVulkanMathEngine::BlobMeanPoolingBackward( const CMeanPoolingDesc&, const CFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

//------------------------------------------------------------------------------------------------------------
// Global max over time pooling

CGlobalMaxOverTimePoolingDesc* CVulkanMathEngine::InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result )
{
	CCommonGlobalMaxOverTimePoolingDesc* desc = new CCommonGlobalMaxOverTimePoolingDesc( source, result );
	return desc;
}

void CVulkanMathEngine::BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& poolingDesc, const CFloatHandle& sourceData,
	const CIntHandle* maxIndicesData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndicesData == 0 );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CCommonGlobalMaxOverTimePoolingDesc& desc = static_cast<const CCommonGlobalMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;

	findMaxValueInColumns( resultData, sourceData, source.BatchLength(), source.BatchWidth() * source.ObjectSize() );
}

void CVulkanMathEngine::BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc&, const CFloatHandle&,
	const CIntHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
