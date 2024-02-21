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

#ifdef NEOML_USE_CUDA

#include <CudaMathEngine.h>
#include <CudaDevice.h>
#include <CudaCommon.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <MathEngineDnnDropout.h>

#include <Kernels/CudaDnnDropoutKernels.h>

namespace NeoML {

CDropoutDesc* CCudaMathEngine::InitDropout(float rate, bool isSpatial, bool isBatchwise)
{
	ASSERT_EXPR(rate >= 0.f && rate < 1.f);
	auto seedDesc = new CSeedDropoutDesc(mathEngine(), false);
	seedDesc->ForwardRate = 1.f - rate;
	seedDesc->IsSpatial = isSpatial;
	seedDesc->IsBatchwise = isBatchwise;
	seedDesc->IsValid = false;
	seedDesc->Value = 1.f / seedDesc->ForwardRate;
	seedDesc->Threshold = (unsigned int)(seedDesc->ForwardRate * UINT_MAX);
	return seedDesc;
}

void CCudaMathEngine::UpdateDropout(CDropoutDesc* dropoutDesc,
	const CBlobDesc* input, const CBlobDesc* output, int seed, bool valid)
{
	auto seedDesc = dynamic_cast<CSeedDropoutDesc*>(dropoutDesc);
	if (seedDesc == nullptr) {
		return;
	}

	seedDesc->IsValid = valid;
	if(valid) {
		seedDesc->Seed = seed;
		updateDesc(seedDesc->Input, input);
		updateDesc(seedDesc->Output, output);
	}
}

void CCudaMathEngine::Dropout( const CDropoutDesc& dropoutDesc,
	const CFloatHandle& inputData, const CFloatHandle& outputData )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( outputData.GetMathEngine() == this );
	SetCudaDevice( device->DeviceNumber );

	const CSeedDropoutDesc& desc = static_cast<const CSeedDropoutDesc&>( dropoutDesc );
	const CBlobDesc& input = *desc.Input;

	if( desc.ForwardRate == 1.f ) {
		VectorCopy( outputData, inputData, input.BlobSize() );
		return;
	}

	const int objectSize = desc.IsSpatial ? input.Channels() : input.ObjectSize();
	const int batchLength = desc.IsBatchwise ? input.ObjectCount() : input.BatchLength();
	const int batchWidth = input.ObjectCount() / batchLength;
	const int maskSize = batchWidth * objectSize;

	if( !desc.IsSpatial ) {
		dim3 blockCount;
		dim3 threadCount;

		getCudaTaskGrid2D(blockCount, threadCount, batchLength, (maskSize + desc.maskAlign - 1) / desc.maskAlign);
		RandomMatrixDropout<<<blockCount, threadCount>>>( GetRaw(inputData), batchLength, maskSize,
			GetRaw(outputData), desc.Seed, desc.ForwardRate );
		return;
	}

	dim3 blockCount;
	dim3 threadCount;

	getCudaTaskGrid3D( blockCount, threadCount, input.ObjectCount(), input.ObjectSize() / objectSize, objectSize );
	RandomSpatialDropout<<<blockCount, threadCount>>>( GetRaw( inputData ), GetRaw( outputData ),
		input.ObjectCount(), input.ObjectSize(), batchWidth, objectSize, desc.Seed, desc.ForwardRate );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
