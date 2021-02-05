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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/NeoMathEngineException.h>
#include <MathEngineDnnLrn.h>
#include <MemoryHandleInternal.h>
#include <CudaMathEngine.h>
#include <CudaDevice.h>
#include <CudaCommon.h>
#include <Kernels/CudaDnnLrnKernels.h>

namespace NeoML {

CLrnDesc* CCudaMathEngine::InitLrn( const CBlobDesc& source, int windowSize, float bias, float alpha, float beta )
{
	return new CMathEngineLrnDesc( source, windowSize, bias, alpha, beta );
}

void CCudaMathEngine::Lrn( const CLrnDesc& lrnDesc, const CConstFloatHandle& input, const CFloatHandle& invSum,
	const CFloatHandle& invSumBeta , const CFloatHandle& output )
{
	ASSERT_EXPR( input.GetMathEngine() == this );
	ASSERT_EXPR( invSum.IsNull() || invSum.GetMathEngine() == this );
	ASSERT_EXPR( invSumBeta.IsNull() || invSumBeta.GetMathEngine() == this );
	ASSERT_EXPR( output.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CMathEngineLrnDesc& desc = static_cast<const CMathEngineLrnDesc&>( lrnDesc );

	const int vectorSize = desc.Source.Channels();
	const int vectorCount = desc.Source.BlobSize() / vectorSize;

	dim3 blockCount;
	dim3 threadCount;
	getCudaTaskGrid2D( blockCount, threadCount, vectorCount, vectorSize );

	const float* inputPtr = GetRaw( input );
	float* outputPtr = GetRaw( output );
	float* invSumPtr = invSum.IsNull() ? outputPtr : GetRaw( invSum );
	float* invSumBetaPtr = invSumBeta.IsNull() ? outputPtr : GetRaw( invSumBeta );

	LrnKernel<<<blockCount, threadCount>>>( inputPtr, invSumPtr, invSumBetaPtr, outputPtr, vectorCount, vectorSize,
		desc.WindowSize, desc.Bias, desc.Alpha, desc.Beta ); 
}

void CCudaMathEngine::LrnBackward( const CLrnDesc& /* desc */, const CConstFloatHandle& /* input */, const CConstFloatHandle& /* output */,
		const CConstFloatHandle& /* outputDiff */, const CConstFloatHandle& /* invSum */, const CConstFloatHandle& /* invSumBeta */,
		const CFloatHandle& /* inputDiff */ )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
