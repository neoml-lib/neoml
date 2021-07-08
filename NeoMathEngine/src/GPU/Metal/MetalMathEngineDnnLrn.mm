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

#ifdef NEOML_USE_METAL

#include <MetalMathEngine.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnLrn.h>
#include <MetalKernel.h>

namespace NeoML {

CLrnDesc* CMetalMathEngine::InitLrn( const CBlobDesc& source, int windowSize, float bias, float alpha, float beta )
{
	return new CMathEngineLrnDesc( source, windowSize, bias, alpha, beta );
}

void CMetalMathEngine::Lrn( const CLrnDesc& lrnDesc, const CConstFloatHandle& input, const CFloatHandle& /* invSum */,
	const CFloatHandle& /* invSumBeta */, const CFloatHandle& output )
{
	ASSERT_EXPR( input.GetMathEngine() == this );
	ASSERT_EXPR( output.GetMathEngine() == this );

	const CMathEngineLrnDesc& desc = static_cast<const CMathEngineLrnDesc&>( lrnDesc );

	const int vectorSize = desc.Source.Channels();
	const int vectorCount = desc.Source.BlobSize() / vectorSize;

	C2DKernel kernel( *queue, "matrixLrn", 1, 1, vectorCount, vectorSize );
	kernel.SetParam( input, 0 );
	kernel.SetParam( output, 1 );
	kernel.SetParam( vectorCount, 2 );
	kernel.SetParam( vectorSize, 3 );
	kernel.SetParam( desc.WindowSize, 4 );
	kernel.SetParam( desc.Bias, 5 );
	kernel.SetParam( desc.Alpha, 6 );
	kernel.SetParam( desc.Beta, 7 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::LrnBackward( const CLrnDesc& /* desc */, const CConstFloatHandle& /* input */, const CConstFloatHandle& /* output */,
		const CConstFloatHandle& /* outputDiff */, const CConstFloatHandle& /* invSum */, const CConstFloatHandle& /* invSumBeta */,
		const CFloatHandle& /* inputDiff */ )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_METAL
