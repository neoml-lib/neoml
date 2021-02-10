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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <VulkanMathEngine.h>
#include <VulkanShader.h>
#include <VulkanDll.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnLrn.h>

#include <shaders/generated/Lrn.h>

namespace NeoML {

CLrnDesc* CVulkanMathEngine::InitLrn( const CBlobDesc& source, int windowSize, float bias, float alpha, float beta )
{
	return new CMathEngineLrnDesc( source, windowSize, bias, alpha, beta );
}

void CVulkanMathEngine::Lrn( const CLrnDesc& lrnDesc, const CConstFloatHandle& input, const CFloatHandle& invSum,
	const CFloatHandle& invSumBeta , const CFloatHandle& output )
{
	const CMathEngineLrnDesc& desc = static_cast<const CMathEngineLrnDesc&>( lrnDesc );
	
	const int vectorSize = desc.Source.Channels();
	const int vectorCount = desc.Source.BlobSize() / vectorSize;
	const size_t bytesTotal = vectorSize * vectorCount * sizeof( float );

	CMemoryHandle bufs[4] = { input, invSum.IsNull() ? output : invSum,
		invSumBeta.IsNull() ? output : invSumBeta, output };
	size_t sizes[4] = { bytesTotal, bytesTotal, bytesTotal, bytesTotal };

	PARAM_STRUCT(Lrn) param = { vectorCount, vectorSize, desc.WindowSize,
		desc.Bias, desc.Alpha, desc.Beta };
	
	runShader( shaderLoader->GET_SHADER_DATA( Lrn, true, 0, 0, 4 ), &param, sizeof( param ), 0, 0, 0, 0, bufs, sizes,
		4, vectorCount, vectorSize, 1 );
}

void CVulkanMathEngine::LrnBackward( const CLrnDesc& /* desc */, const CConstFloatHandle& /* input */, const CConstFloatHandle& /* output */,
		const CConstFloatHandle& /* outputDiff */, const CConstFloatHandle& /* invSum */, const CConstFloatHandle& /* invSumBeta */,
		const CFloatHandle& /* inputDiff */ )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif  // NEOML_USE_VULKAN
