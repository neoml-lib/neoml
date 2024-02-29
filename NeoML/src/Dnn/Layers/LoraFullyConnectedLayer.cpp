/* Copyright © 2023 ABBYY

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

#include <NeoML/Dnn/Layers/LoraFullyConnectedLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

//----------------------------------------------------------------------------------------------

CLoraFullyConnectedLayer::CLoraFullyConnectedLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, ( name != nullptr ) ? name : "CDnnLoraFullyConnectedLayer", /*isLearnable*/true )
{
	paramBlobs.SetSize( 2 );

	CLoraParams defaultParams;
	initialize( defaultParams );
}

CLoraFullyConnectedLayer::CLoraFullyConnectedLayer( CDnnBlob& baseWeights, CDnnBlob* baseFreeTerms,
		const CLoraParams& params ) :
	CBaseLayer( baseWeights.GetMathEngine(), "CDnnLoraFullyConnectedLayer", /*isLearnable*/true )
{
	paramBlobs.SetSize( 2 );

	initialize( params );

	weightsBase = &baseWeights;
	freeTermsBase = baseFreeTerms;
}

CLoraFullyConnectedLayer::~CLoraFullyConnectedLayer()
{
	destroyDropoutDesc();
}

void CLoraFullyConnectedLayer::initDropoutDesc()
{
	if( desc == nullptr ) {
		desc = MathEngine().InitDropout( lora.Dropout, /*isSpatial*/false, /*isBatchwise*/false,
			inputBlobs[0]->GetDesc(), inputBlobs[0]->GetDesc(), GetDnn()->Random().Next() );
	}
}

void CLoraFullyConnectedLayer::destroyDropoutDesc()
{
	if( desc != nullptr ) {
		delete desc;
		desc = nullptr;
	}
}

static const int LoraFullyConnectedLayerVersion = 0;

void CLoraFullyConnectedLayer::Serialize( CArchive& archive )
{
	( void ) archive.SerializeVersion( LoraFullyConnectedLayerVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( isMerged );
	CLoraParams params = lora;
	params.Serialize( archive );

	SerializeBlob( MathEngine(), archive, weightsBase );
	SerializeBlob( MathEngine(), archive, freeTermsBase );

	if( archive.IsLoading() ) {
		initialize( params );
		destroyDropoutDesc();
	}
}

void CLoraFullyConnectedLayer::UpdateParams( const CLoraParams& newParams, CDnnBlob* newA, CDnnBlob* newB )
{
	split();

	initialize( newParams );

	WeightsA() = newA;
	WeightsB() = newB;
}

void CLoraFullyConnectedLayer::Reshape()
{
	CheckLayerArchitecture( GetInputCount() == 1, "LoraFullyConnected Layer must have only 1 input" );
	CheckLayerArchitecture( GetOutputCount() == 1, "LoraFullyConnected Layer must have only 1 output" );

	if( IsBackwardPerformed() || IsLearningPerformed() ) {
		split();
	} else {
		merge();
	}

	NeoAssert( weightsBase != nullptr );

	if( WeightsA() == nullptr ) { // Create A weights matrix
		CBlobDesc ADesc = inputDescs[0];
		ADesc.SetDimSize( BD_BatchLength, 1 );
		ADesc.SetDimSize( BD_BatchWidth, lora.Rank ); // A^T size : Rank x In.Size
		ADesc.SetDimSize( BD_ListSize, 1 );
		WeightsA() = CDnnBlob::CreateBlob( MathEngine(), CT_Float, ADesc );
		// Initialize Xavier (default)
		InitializeParamBlob( 0, *WeightsA() );
	}
	if( WeightsB() == nullptr ) { // Create B weights matrix
		CBlobDesc BDesc = inputDescs[0];
		BDesc.SetDimSize( BD_BatchLength, 1 );
		BDesc.SetDimSize( BD_BatchWidth, NumberOfElements() ); // B^T size : NumberOfElements x Rank
		BDesc.SetDimSize( BD_ListSize, 1 );
		BDesc.SetDimSize( BD_Height, 1 );
		BDesc.SetDimSize( BD_Width, 1 );
		BDesc.SetDimSize( BD_Depth, 1 );
		BDesc.SetDimSize( BD_Channels, lora.Rank );
		WeightsB() = CDnnBlob::CreateBlob( MathEngine(), CT_Float, BDesc );
		// Initialize zeroes
		WeightsB()->Clear();
	}
	// For each layer element there is a channel in the output blob
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize( BD_Height, 1 );
	outputDescs[0].SetDimSize( BD_Width, 1 );
	outputDescs[0].SetDimSize( BD_Depth, 1 );
	outputDescs[0].SetDimSize( BD_Channels, NumberOfElements() );

	destroyDropoutDesc();
}

void CLoraFullyConnectedLayer::RunOnce()
{
	CConstFloatHandle inputData = inputBlobs[0]->GetData();
	const int inputHeight = inputBlobs[0]->GetObjectCount();
	const int inputWidth = inputBlobs[0]->GetObjectSize();
	const int inputSize = inputBlobs[0]->GetDataSize();

	CFloatHandle outputData = outputBlobs[0]->GetData();
	const int outputHeight = outputBlobs[0]->GetObjectCount();
	const int outputWidth = outputBlobs[0]->GetObjectSize();

	CConstFloatHandle weightsData = weightsBase->GetData();
	const int weightsHeight = NumberOfElements();
	const int weightsWidth = weightsBase->GetObjectSize();

	NeoPresume( inputWidth == weightsWidth );
	NeoPresume( outputWidth == weightsHeight );
	NeoPresume( outputHeight == inputHeight );

	const int BHeight = WeightsB()->GetObjectCount();
	const int tempAxBSize = inputHeight * BHeight;

	const bool inference = !IsBackwardPerformed() && !IsLearningPerformed();
	CFloatHandleStackVar temp( MathEngine(), inference ? 0 : max( inputSize, tempAxBSize ) );
	CFloatHandle tempAxB = temp.GetHandle();

	/*          +---------+   +--------+                 \  
	*        (x)|  BASE^T |(+)|FreeTerm|                  \  
	*  +----+   +---------+   +--------+                  |       +-----+    
	*  | IN |                                             |(+) (=)| OUT |
	*  +----+   +---------+   +-----+   +-----+           |       +-----+
	*        (x)| Dropout |-->| A^T |(x)| B^T |(x)(scale) /     
	*           +---------+   +-----+   +-----+          /
	*/
	if( inference ) {
		NeoPresume( IsMerged() );
	} else {
		NeoPresume( !IsMerged() );
		// +----+   +---------+   +-----+   +-----+             +-------+
		// | IN |(x)| Dropout |-->| A^T |(x)| B^T |(x)(scale)(=)|TMP OUT|
		// +----+   +---------+   +-----+   +-----+             +-------+

		const int AHeight = WeightsA()->GetObjectCount();
		const int tempInputASize = inputHeight * AHeight;

		CConstFloatHandle tempInputData = inputData;
		if( lora.Dropout > 0.f ) {
			initDropoutDesc();
			MathEngine().Dropout( *desc, inputBlobs[0]->GetData(), temp.GetHandle() );
			tempInputData = temp.GetHandle();
		}

		NeoPresume( outputBlobs[0]->GetDataSize() >= tempInputASize );
		CFloatHandle tempInputA = outputData;
		MathEngine().MultiplyMatrixByTransposedMatrix( /*batchSize*/1,
			/*first*/tempInputData, inputHeight, inputWidth,
			/*second*/WeightsA()->GetData(), AHeight,
			/*result*/tempInputA, tempInputASize );

		if( scaling->GetData().GetValue() != 1.f ) {
			MathEngine().VectorMultiply( tempInputA, tempInputA, tempInputASize, scaling->GetData() );
		}

		MathEngine().MultiplyMatrixByTransposedMatrix( /*batchSize*/1,
			/*first*/tempInputA, inputHeight, AHeight,
			/*second*/WeightsB()->GetData(), BHeight,
			/*result*/tempAxB, tempAxBSize );
	}

	{
		// +----+   +--------+   +----------+   +-----+
		// | IN |(x)| BASE^T |(+)| FreeTerm |(=)| OUT | 
		// +----+   +--------+   +----------+   +-----+
		MathEngine().MultiplyMatrixByTransposedMatrix(
			/*first*/inputData, inputHeight, inputWidth, inputWidth,
			/*second*/weightsData, weightsHeight, weightsWidth,
			/*result*/outputData, outputWidth, /*unused*/0 );

		if( freeTermsBase != nullptr ) {
			MathEngine().AddVectorToMatrixRows( /*batchSize*/1, outputData,
				outputData, outputHeight, outputWidth, freeTermsBase->GetData() );
		}
	}

	if ( !inference ) {
		// +-----+  +-------+
		// | OUT |+=|TMP OUT|
		// +-----+  +-------+
		NeoPresume( outputBlobs[0]->GetDataSize() == tempAxBSize );
		MathEngine().VectorAdd( outputData, tempAxB, outputData, tempAxBSize );
	}
}

void CLoraFullyConnectedLayer::BackwardOnce()
{
	NeoPresume( !IsMerged() );

	CFloatHandle inputDiff = inputDiffBlobs[0]->GetData();
	const int inputDiffSize = inputDiffBlobs[0]->GetDataSize();

	CConstFloatHandle outputDiffData = outputDiffBlobs[0]->GetData();
	const int outputDiffHeight = outputDiffBlobs[0]->GetObjectCount();
	const int outputDiffWidth = outputDiffBlobs[0]->GetObjectSize();

	CConstFloatHandle weightsData = weightsBase->GetData();
	const int weightsWidth = weightsBase->GetObjectSize();

	const int BWidth = WeightsB()->GetObjectSize();
	const int tempBDiffSize = outputDiffHeight * BWidth;
	NeoPresume( inputDiffSize == outputDiffHeight * weightsWidth );
	CFloatHandleStackVar temp( MathEngine(), max( inputDiffSize, tempBDiffSize ) );

	CFloatHandle tempBDiff = temp.GetHandle();
	MathEngine().MultiplyMatrixByMatrix( /*batchSize*/1,
		/*first*/outputDiffData, outputDiffHeight, outputDiffWidth,
		/*second*/WeightsB()->GetData(), BWidth,
		/*result*/tempBDiff, tempBDiffSize );

	if( scaling->GetData().GetValue() != 1.f ) {
		MathEngine().VectorMultiply( tempBDiff, tempBDiff, tempBDiffSize, scaling->GetData() );
	}

	const int AWidth = WeightsA()->GetObjectSize();
	const int tempADiffSize = outputDiffHeight * AWidth;
	NeoPresume( tempADiffSize == inputDiffSize );

	MathEngine().MultiplyMatrixByMatrix( /*batchSize*/1,
		/*first*/tempBDiff, outputDiffHeight, BWidth,
		/*second*/WeightsA()->GetData(), AWidth,
		/*result*/inputDiff, tempADiffSize );

	CFloatHandle tempInputDiff = temp.GetHandle();
	const bool dropout = lora.Dropout > 0.f;
	if( dropout ) {
		NeoAssert( desc != nullptr ); // Backward pass is only possible when learning
		MathEngine().Dropout( *desc, inputDiff, tempInputDiff );
		if( !GetDnn()->IsRecurrentMode() || GetDnn()->IsFirstSequencePos() ) {
			destroyDropoutDesc(); // Clear the memory after the whole sequence is processed
		}
	}

	CFloatHandle resultDiff = dropout ? inputDiff : tempInputDiff;
	MathEngine().MultiplyMatrixByMatrix( /*batchSize*/1,
		/*first*/outputDiffData, outputDiffHeight, outputDiffWidth,
		/*second*/weightsData, weightsWidth,
		/*result*/resultDiff, inputDiffSize );

	MathEngine().VectorAdd( tempInputDiff, inputDiff, inputDiff, inputDiffSize );
}

void CLoraFullyConnectedLayer::LearnOnce()
{
	NeoPresume( !IsMerged() );

	CConstFloatHandle outputDiffData = outputDiffBlobs[0]->GetData();
	const int outputDiffHeight = outputDiffBlobs[0]->GetObjectCount();
	const int outputDiffWidth = outputDiffBlobs[0]->GetObjectSize();

	CConstFloatHandle inputData = inputBlobs[0]->GetData();
	const int inputHeight = inputBlobs[0]->GetObjectCount();
	const int inputWidth = inputBlobs[0]->GetObjectSize();
	const int BWidth = WeightsB()->GetObjectSize();
	const int AHeight = WeightsA()->GetObjectCount();

	const int tempBDiffSize = outputDiffHeight * BWidth;
	const int tempInputASize = inputHeight * AHeight;
	CFloatHandleStackVar temp( MathEngine(), max( tempBDiffSize, tempInputASize ) );

	{
		CFloatHandle tempBDiff = temp.GetHandle();
		// Prepare Diff
		MathEngine().MultiplyMatrixByMatrix( /*batchSize*/1,
			/*first*/outputDiffData, outputDiffHeight, outputDiffWidth,
			/*second*/WeightsB()->GetData(), BWidth,
			/*result*/tempBDiff, tempBDiffSize );

		if( scaling->GetData().GetValue() != 1.f ) {
			MathEngine().VectorMultiply( tempBDiff, tempBDiff, tempBDiffSize, scaling->GetData() );
		}

		const int AWidth = WeightsADiff()->GetObjectSize();
		const int ASize = WeightsADiff()->GetDataSize();
		NeoPresume( AWidth == inputWidth );
		// Apply Diff for A Weights
		MathEngine().MultiplyTransposedMatrixByMatrixAndAdd(
			/*first*/tempBDiff, outputDiffHeight, BWidth, BWidth,
			/*second*/inputData, inputWidth, inputWidth,
			/*result*/WeightsADiff()->GetData(), AWidth, ASize );
	}

	{
		CFloatHandle tempInputA = temp.GetHandle();
		// Prepare Diff
		MathEngine().MultiplyMatrixByTransposedMatrix( /*batchSize*/1,
			/*first*/inputData, inputHeight, inputWidth,
			/*second*/WeightsA()->GetData(), AHeight,
			/*result*/tempInputA, tempInputASize );

		const int BSize = WeightsBDiff()->GetDataSize();
		// Apply Diff for A Weights
		MathEngine().MultiplyTransposedMatrixByMatrixAndAdd(
			/*first*/outputDiffData, outputDiffHeight, outputDiffWidth, outputDiffWidth,
			/*second*/tempInputA, AHeight, AHeight,
			/*result*/WeightsBDiff()->GetData(), BWidth, BSize );
	}
}

void CLoraFullyConnectedLayer::initialize( const CLoraParams& params )
{
	NeoAssert( params.Dropout >= 0.f && params.Dropout < 1.f );
	NeoAssert( params.Rank > 0 );
	NeoAssert( params.Alpha > 0.f );

	lora = params;
	const float scale = lora.Alpha / lora.Rank;
	NeoAssert( scale > 0.f );

	if( scaling == nullptr ) {
		scaling = CDnnBlob::CreateVector( MathEngine(), CT_Float, 1 );
	}
	scaling->GetData().SetValue( scale );
}

void CLoraFullyConnectedLayer::merge()
{
	if( IsMerged() ) {
		return;
	}

	isMerged = true;
	recalcBaseWeights();
}

void CLoraFullyConnectedLayer::split()
{
	if( !IsMerged() ) {
		return;
	}

	isMerged = false;
	recalcBaseWeights();
}

void CLoraFullyConnectedLayer::recalcBaseWeights()
{
	// isMerged is a newly changed state
	if( WeightsA() == nullptr ) {
		NeoAssert( WeightsB() == nullptr );
		// weights A and B weren't initalized
		// the weights will be initialized in a way that untrained LoRA won't affect base weights
		// as a results we can relax for now
		return;
	}

	const int inputSize = WeightsA()->GetObjectSize();
	const int outputSize = NumberOfElements();
	const int bSize = lora.Rank * outputSize;
	NeoAssert( WeightsB()->GetDataSize() == bSize );

	CFloatHandleStackVar temp( MathEngine(), static_cast<size_t>( bSize + 1 ) );
	CFloatHandle bTransposed = temp.GetHandle();
	CFloatHandle mult = temp.GetHandle() + bSize;

	MathEngine().TransposeMatrix( /*batchSize*/1, WeightsB()->GetData(),
		/*h*/outputSize, /*mid*/1, /*w*/lora.Rank, /*channels*/1, bTransposed, bSize );

	// during split we must substract A*B from merged weights
	const float scale = scaling->GetData().GetValue();
	const float multValue = IsMerged() ? scale : -scale;
	if( multValue != 1 ) {
		mult.SetValue( multValue );
		MathEngine().VectorMultiply( bTransposed, bTransposed, bSize, mult );
	}

	MathEngine().MultiplyTransposedMatrixByMatrixAndAdd(
		/*first*/bTransposed, lora.Rank, outputSize, outputSize,
		/*second*/WeightsA()->GetData(), inputSize, inputSize,
		/*result*/weightsBase->GetData(), inputSize, inputSize * outputSize );
}

} // namespace NeoML
