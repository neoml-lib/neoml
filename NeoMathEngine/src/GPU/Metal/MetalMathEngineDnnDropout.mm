/* Copyright © 2017-2024 ABBYY Production LLC

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
#include <MathEngineDnnDropout.h>
#include <MathEngineCommon.h>
#include <MetalKernel.h>

namespace NeoML {

CDropoutDesc* CMetalMathEngine::InitDropout(float rate, bool isSpatial, bool isBatchwise)
{
	ASSERT_EXPR(rate >= 0.f && rate < 1.f);
	auto maskDesc = new CMaskDropoutDesc();
	maskDesc->ForwardRate = 1.f - rate;
	maskDesc->IsSpatial = isSpatial;
	maskDesc->IsBatchwise = isBatchwise;
	maskDesc->IsValid = false;
	maskDesc->Value = 1.f / maskDesc->ForwardRate;
	maskDesc->Threshold = (unsigned int)(maskDesc->ForwardRate * UINT_MAX);
	return maskDesc;
}

void CMetalMathEngine::UpdateDropout(CDropoutDesc* dropoutDesc, const CBlobDesc* input, const CBlobDesc* output,
	int seed, bool valid)
{
	auto maskDesc = dynamic_cast<CMaskDropoutDesc*>(dropoutDesc);
	if(maskDesc == nullptr) {
		return;
	}

	maskDesc->IsValid = valid;
	if(maskDesc->Mask != nullptr) {
		delete maskDesc->Mask;
		maskDesc->Mask = nullptr;
	}
	
	if(valid) {
		updateDesc(maskDesc->Input, input);
		updateDesc(maskDesc->Output, output);

		maskDesc->Seed = seed;
		maskDesc->Mask = new CFloatHandleVar(mathEngine(), getMaskSize(maskDesc->IsSpatial, maskDesc->IsBatchwise, *input));
		mathEngine().VectorFillBernoulli(maskDesc->Mask->GetHandle(), maskDesc->ForwardRate, maskDesc->Mask->Size(),
			1.f / maskDesc->ForwardRate, maskDesc->Seed);
	}
}

void CMetalMathEngine::Dropout( const CDropoutDesc& dropoutDesc, const CFloatHandle& inputData, const CFloatHandle& outputData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( outputData.GetMathEngine() == this );

	const CMaskDropoutDesc& desc = static_cast<const CMaskDropoutDesc&>( dropoutDesc );
	const CBlobDesc& input = *(desc.Input);
	const CBlobDesc& output = *(desc.Output);

	if( desc.ForwardRate == 1.f ) {
		VectorCopy( outputData, inputData, input.BlobSize() );
		return;
	}

	const int objectSize = desc.IsSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = desc.IsBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;
	const int maskSize = batchWidth * objectSize;

	ASSERT_EXPR( desc.Mask->Size() == maskSize );

	if( !desc.IsSpatial ) {
		MultiplyMatrixByDiagMatrix( inputData, batchLength, maskSize, desc.Mask, outputData, output.BlobSize() );
		return;
	}

	const int maskObjectSize = maskSize / batchWidth;

	C3DKernel kernel( *queue, "cubeKernelChannelLastBlobSpatialDropout", 1, 1, 1, input.ObjectCount(),
		input.ObjectSize() / maskObjectSize, maskObjectSize );
	kernel.SetParam( inputData, 0 );
	kernel.SetParam( desc.Mask, 1 );
	kernel.SetParam( outputData, 2 );
	kernel.SetParam( input.ObjectCount(), 3 );
	kernel.SetParam( input.ObjectSize(), 4 );
	kernel.SetParam( batchWidth, 5 );
	kernel.SetParam( maskObjectSize, 6 );
	ASSERT_EXPR( kernel.Run() );
}

} // namespace NeoML

#endif // NEOML_USE_METAL
