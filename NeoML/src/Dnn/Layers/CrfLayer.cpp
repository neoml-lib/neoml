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

#include <NeoML/Dnn/Layers/CrfLayer.h>
#include <NeoML/Dnn/Layers/SequenceSumLayer.h>
#include <NeoML/Dnn/Layers/SubSequenceLayer.h>
#include <NeoML/TraditionalML/VariableMatrix.h>
#include <NeoML/Dnn/Layers/DropoutLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>

namespace NeoML {

static const char* DropOutName = "DropOut";

CCrfCalculationLayer::CCrfCalculationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnCrfCalculationLayer", true ),
	paddingClass( 0 ),
	doCalculateBestPrevClass( false )
{
	paramBlobs.SetSize(1);
}

void CCrfCalculationLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() >= 2 && GetInputCount() == GetOutputCount(),
		GetName(), "CRF layer with incorrect numbers of input and output" );
	for(int i = 1; i < GetInputCount(); i++) {
		CheckArchitecture( inputDescs[I_ClassLogProb].BatchLength() == inputDescs[i].BatchLength()
			&& inputDescs[I_ClassLogProb].BatchWidth() == inputDescs[i].BatchWidth(),
			GetName(), CString("incorrect batch size at input " + Str(i)) );
	}
	int numberOfClasses = inputDescs[I_ClassLogProb].ObjectSize();
	// Create a transition matrix
	if(Transitions() == 0) {
		Transitions() = CDnnBlob::CreateMatrix(MathEngine(), CT_Float, numberOfClasses, numberOfClasses);
		// Initialize the transition matrix
		InitializeParamBlob(0, *Transitions());
	} else {
		CheckArchitecture( Transitions()->DimSize(0) == numberOfClasses,
			GetName(), "transition table size is not equal to number of classes" );
	}
	// Create output blobs
	// The optimal class sequence
	outputDescs[O_BestPrevClass] = CBlobDesc( CT_Int );
	outputDescs[O_BestPrevClass].SetDimSize( BD_BatchLength, inputDescs[I_ClassLogProb].BatchLength() );
	outputDescs[O_BestPrevClass].SetDimSize( BD_BatchWidth, inputDescs[I_ClassLogProb].BatchWidth() );
	outputDescs[O_BestPrevClass].SetDimSize( BD_Channels, numberOfClasses );
	// The partial function value in this sequence position
	// (alpha in the forward-backward algorithm, forward pass)
	// Equals the sum of estimates (logits) of all sequences that stop at this position
	outputDescs[O_ClassSeqLogProb] = outputDescs[O_BestPrevClass];
	outputDescs[O_ClassSeqLogProb].SetDataType( CT_Float );
	
	// Create a temporary blob
	tempSumBlob = CDnnBlob::Create2DImageBlob(MathEngine(), CT_Float, inputDescs[I_ClassLogProb].BatchLength(),
		inputDescs[I_ClassLogProb].BatchWidth(), numberOfClasses, numberOfClasses, 1);
	RegisterRuntimeBlob(tempSumBlob);

	if(GetInputCount() > 2) {
		CheckArchitecture( inputDescs[I_Label].GetDataType() == CT_Int, GetName(), "labels should have the integer type" );
		// The estimate (logit) of the correct class in this position
		outputDescs[O_LabelLogProb] = outputDescs[O_ClassSeqLogProb];
		outputDescs[O_LabelLogProb].SetDimSize( BD_Channels, 1 );
	}
	NeoAssert(GetPaddingClass() < numberOfClasses);
}

// Gets the previous labels for the correct sequences
CPtr<CDnnBlob> CCrfCalculationLayer::getPrevLabels() const
{
	// The labels should either be a child blob or an independent blob, but in any case have BatchLength equal to 1
	NeoAssert( inputBlobs[I_Label]->GetBatchLength() == 1 );
	// There are no previous labels for the first blob in sequence, or if the sequence length is only 1
	NeoAssert( inputBlobs[I_Label]->GetParent() != 0 && inputBlobs[I_Label]->GetParentPos() > 0 );

	if( prevLabels == 0 || inputBlobs[I_Label]->GetParent() != prevLabels->GetParent() ) {
		// The input blob has changed, need to update previous labels
		prevLabels = CDnnBlob::CreateWindowBlob(inputBlobs[I_Label]->GetParent(), 1);
	}
	// The previous labels could be found by shifting one step back from the current position in the parent blob inputBlobs
	prevLabels->SetParentPos(inputBlobs[I_Label]->GetParentPos() - 1);
	return prevLabels;
}

// Calculate the correct class probability
void CCrfCalculationLayer::calcLabelProbability()
{
	int batchWidth = inputBlobs[I_ClassLogProb]->GetBatchWidth();
	int numberOfClasses = inputBlobs[I_ClassLogProb]->GetObjectSize();
	outputBlobs[O_LabelLogProb]->Clear();
	// Add the unary estimates (the fully-connected layer output for this element features)
	MathEngine().AddMatrixElementsToVector(inputBlobs[I_ClassLogProb]->GetData(), batchWidth, 
		numberOfClasses, inputBlobs[I_Label]->GetData<int>(), 
		outputBlobs[O_LabelLogProb]->GetData(), outputBlobs[O_LabelLogProb]->GetDataSize());

	if( !isFirstStep() ) {
		// Add the binary estimates (of transitioning from the previous correct classes to the current ones)
		MathEngine().AddMatrixElementsToVector( Transitions()->GetData(), numberOfClasses, numberOfClasses,
			inputBlobs[I_Label]->GetData<int>(), getPrevLabels()->GetData<int>(),
			outputBlobs[O_LabelLogProb]->GetData(), outputBlobs[O_LabelLogProb]->GetDataSize() );
	}
}

// Checks if we are at the first (possibly the only) step
bool CCrfCalculationLayer::isFirstStep() const
{
	return !GetDnn()->IsRecurrentMode() || GetDnn()->IsFirstSequencePos();
}

void CCrfCalculationLayer::RunOnce()
{
	// The unary estimates of the current elements (the fully-connected layer output)
	CConstFloatHandle currentProbabilities = inputBlobs[I_ClassLogProb]->GetData();
	// Always clear tempSumBlob so it is not left uninitialized
	tempSumBlob->Clear();

	if( isFirstStep() || ( IsLearningPerformed() && !doCalculateBestPrevClass ) ) {
		// We don't compute O_BestPrevClass output at the first step and also during training when not asked explicitly.
		// Clear the O_BestPrevClass output so it is not left uninitialized.
		outputBlobs[O_BestPrevClass]->Clear();
	}

	if( isFirstStep() ) {
		// At the first step all we have to do is initialize the O_ClassSeqLogProb output with current element probabilities
		MathEngine().VectorCopy( outputBlobs[O_ClassSeqLogProb]->GetData(), currentProbabilities,
			outputBlobs[O_ClassSeqLogProb]->GetDataSize() );
	} else {
		int batchWidth = inputBlobs[I_ClassLogProb]->GetBatchWidth();
		int numberOfClasses = inputBlobs[I_ClassLogProb]->GetObjectSize();

		// The vector of partial function value at the previous step (alpha in the forward-backward algorithm, forward pass)
		CConstFloatHandle sequenceProbabilities = inputBlobs[I_ClassSeqLogProb]->GetData();

		CFloatHandle tempSum = tempSumBlob->GetData();

		// Replicate the transitions matrix batchWidth times (manual broadcast)
		// Add the binary estimates of transition from any previous step to the current one
		MathEngine().AddVectorToMatrixRows(1, tempSum, tempSum, batchWidth,
			numberOfClasses * numberOfClasses, Transitions()->GetData());
		// Add the partial function values calculated above
		MathEngine().AddVectorToMatrixRows(batchWidth, tempSum, tempSum,
			numberOfClasses, numberOfClasses, sequenceProbabilities);

		if(!IsLearningPerformed()) {
			// Back-pointers for the Viterbi algorithm
			MathEngine().FindMaxValueInRows(tempSum, numberOfClasses * batchWidth, numberOfClasses, 
				outputBlobs[O_ClassSeqLogProb]->GetData(), outputBlobs[O_BestPrevClass]->GetData<int>(),
				outputBlobs[O_ClassSeqLogProb]->GetDataSize());
		} else {
			// Algorithm forward pass (calculate the new alphas - the partial function values at this step):
			MathEngine().MatrixLogSumExpByRows( tempSum, numberOfClasses  * batchWidth, numberOfClasses,
				outputBlobs[O_ClassSeqLogProb]->GetData(), outputBlobs[O_ClassSeqLogProb]->GetDataSize() );
			// Calculate O_BestPrevClass output, if required
			if( doCalculateBestPrevClass ) {
				// Update the throw-away blob dimensions so it can contain the required data
				if( discardedBestPrevClassMax == 0 || !discardedBestPrevClassMax->HasEqualDimensions( outputBlobs[O_ClassSeqLogProb] ) ) {
					discardedBestPrevClassMax = outputBlobs[O_ClassSeqLogProb]->GetClone();
				}
				// Back-pointers for the Viterbi algorithm during training.
				// We use a dummy blob for the by-product max values because during training O_ClassSeqLogProb must contain the result of log-sum-exp and not plain max values.
				MathEngine().FindMaxValueInRows(tempSum, numberOfClasses * batchWidth, numberOfClasses, 
					discardedBestPrevClassMax->GetData(), outputBlobs[O_BestPrevClass]->GetData<int>(), outputBlobs[O_BestPrevClass]->GetDataSize());
			}
		}
		// Add unary estimates at current step (log sum exp will take place at the next step)
		MathEngine().VectorAdd( currentProbabilities, outputBlobs[O_ClassSeqLogProb]->GetData(),
			outputBlobs[O_ClassSeqLogProb]->GetData(), outputBlobs[O_ClassSeqLogProb]->GetDataSize() );
	}

	// Calculate the correct class probability
	if( GetOutputCount() > 2 ) {
		calcLabelProbability();
	}
}

void CCrfCalculationLayer::BackwardOnce()
{
	int numberOfClasses = inputBlobs[I_ClassLogProb]->GetObjectSize();
	int batchWidth = inputBlobs[0]->GetBatchWidth();

	// Input #0 is the fully-connected layer output (logarithm of the probability of each class in this position)
	inputDiffBlobs[I_ClassLogProb]->CopyFrom(outputDiffBlobs[O_ClassSeqLogProb]);
	MathEngine().AddVectorToMatrixElements(inputDiffBlobs[I_ClassLogProb]->GetData(), batchWidth, 
		numberOfClasses, inputBlobs[I_Label]->GetData<int>(), outputDiffBlobs[O_LabelLogProb]->GetData());

	if( !isFirstStep() ) {
		// Input #1 contains the feedback from the output #1
		CFloatHandle tempSum = tempSumBlob->GetData();
		MathEngine().MatrixSoftmaxByRows(tempSum, numberOfClasses * batchWidth, numberOfClasses, tempSum);
		MathEngine().MultiplyMatrixByMatrix(batchWidth,
			outputDiffBlobs[O_ClassSeqLogProb]->GetData(), 1, numberOfClasses, 
			tempSum, numberOfClasses,
			inputDiffBlobs[I_ClassSeqLogProb]->GetData(), 
			inputDiffBlobs[I_ClassSeqLogProb]->GetDataSize());
	}
}

void CCrfCalculationLayer::LearnOnce()
{
	if( isFirstStep() ) {
		// At the first step there are no transitions, so nothing to learn
		return;
	}

	int numberOfClasses = inputBlobs[I_ClassLogProb]->GetObjectSize();
	int batchWidth = inputBlobs[I_ClassLogProb]->GetBatchWidth();

	// Output #1 contains sequence probabilities
	MathEngine().MultiplyDiagMatrixByMatrixAndAdd(batchWidth,
		outputDiffBlobs[O_ClassSeqLogProb]->GetData(),
		numberOfClasses, tempSumBlob->GetData(), numberOfClasses, TransitionsDiff()->GetData());
	// Output #2 contains the correct class probability
	MathEngine().AddVectorToMatrixElements( TransitionsDiff()->GetData(), numberOfClasses,
		numberOfClasses, inputBlobs[I_Label]->GetData<int>(), getPrevLabels()->GetData<int>(),
		outputDiffBlobs[O_LabelLogProb]->GetData(), outputDiffBlobs[O_LabelLogProb]->GetDataSize() );
}

static const int CrfCalculationLayerVersion = 2001;

void CCrfCalculationLayer::Serialize( CArchive& archive )
{
	int version = archive.SerializeVersion( CrfCalculationLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( paddingClass );
	if( version >= 2001 ) {
		archive.Serialize( doCalculateBestPrevClass );
	}
}

///////////////////////////////////////////////////////////////////////////////////
// CCrfInternalLossLayer

void CCrfInternalLossLayer::BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize,
	CConstFloatHandle label,  int labelSize, CFloatHandle lossValue, CFloatHandle dataLossGradient,
	CFloatHandle labelLossGradient)
{
	NeoAssert(labelSize == 1);

	// The loss function is the correct sequence logarithm with the minus sign
	CFloatHandleStackVar logZ(MathEngine(), batchSize);
	// The data vector contains the partial function values for different labels at the end of the sequence
	// Here we calculate the last total over all labels and get the final value of the partial function
	MathEngine().MatrixLogSumExpByRows(data, batchSize, vectorSize, logZ, batchSize);
	// Calculate the loss. Divide the correct sequence probability by the total estimate for all sequences
	// This will be equivalent to the logarithms difference in the log space
	MathEngine().VectorSub(logZ, label, lossValue, batchSize);

	if(!dataLossGradient.IsNull()) {
		MathEngine().MatrixSoftmaxByRows(data, batchSize, vectorSize, dataLossGradient);
	}
	if(!labelLossGradient.IsNull()) {
		MathEngine().VectorFill(labelLossGradient, -1.f, batchSize);
	}
}

static const int CrfInternalLossLayerVersion = 2000;

void CCrfInternalLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CrfInternalLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );
}

///////////////////////////////////////////////////////////////////////////////////
// CCrfLossLayer

CCrfLossLayer::CCrfLossLayer( IMathEngine& mathEngine ) :
	CCompositeLayer( mathEngine, "CCnnCrfLossLayer" )
{
	buildLayer();
}

void CCrfLossLayer::buildLayer()
{
	CPtr<CSubSequenceLayer> sequenceProbabilities = FINE_DEBUG_NEW CSubSequenceLayer( MathEngine() );
	// The last sequence element
	sequenceProbabilities->SetLength(1);
	sequenceProbabilities->SetStartPos(-1);
	AddLayer(*sequenceProbabilities);
	SetInputMapping(I_ClassSeqLogProb, *sequenceProbabilities);

	CPtr<CSequenceSumLayer> labelProbabilities = FINE_DEBUG_NEW CSequenceSumLayer( MathEngine() );
	AddLayer(*labelProbabilities);
	SetInputMapping(I_LabelLogProb, *labelProbabilities);

	internalLossLayer = FINE_DEBUG_NEW CCrfInternalLossLayer( MathEngine() );
	AddLayer(*internalLossLayer);
	internalLossLayer->Connect(0, *sequenceProbabilities);
	internalLossLayer->Connect(1, *labelProbabilities);
	
	// The dummy sink for network building
	CPtr<CSinkLayer> dummySink = FINE_DEBUG_NEW CSinkLayer( MathEngine() );
	AddLayer(*dummySink);
	SetInputMapping(I_BestPrevClass, *dummySink);
}

static const int CrfLossLayerVersion = 2000;

void CCrfLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CrfLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CCompositeLayer::Serialize( archive );
	
	if( archive.IsLoading() ) {
		internalLossLayer = CheckCast<CCrfInternalLossLayer>( GetLayer( internalLossLayer->GetName() ) );
	}
}

///////////////////////////////////////////////////////////////////////////////////

CCrfLayer::CCrfLayer( IMathEngine& mathEngine ) :
	CRecurrentLayer( mathEngine, "CCnnCrfLayer" )
{
	buildLayer(0.f);
}

// Builds the layer
void CCrfLayer::buildLayer(float dropOut)
{
	// Initialize the back link
	if(backLink == 0) {
		backLink = FINE_DEBUG_NEW CBackLinkLayer( MathEngine() );
	}
	AddBackLink(*backLink);

	if(hiddenLayer == 0) {
		hiddenLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	}
	// Get the element features
	SetInputMapping(I_Features, *hiddenLayer);
	AddLayer(*hiddenLayer);
	// Create a dropout layer if needed
	if(dropOut > 0) {
		dropOutLayer = FINE_DEBUG_NEW CDropoutLayer( MathEngine() );
		dropOutLayer->SetName(DropOutName);
		dropOutLayer->SetDropoutRate(dropOut);
		dropOutLayer->Connect(*hiddenLayer);
		AddLayer(*dropOutLayer);
	} else {
		dropOutLayer = 0;
	}
	// Create a CRF calculation layer
	if(calculator == 0) {
		calculator = FINE_DEBUG_NEW CCrfCalculationLayer( MathEngine() );
	}
	AddLayer(*calculator);
	// Get the non-normalized logarithm of class probability as an input
	if(dropOutLayer == 0) {
		calculator->Connect(CCrfCalculationLayer::I_ClassLogProb, *hiddenLayer);
	} else {
		calculator->Connect(CCrfCalculationLayer::I_ClassLogProb, *dropOutLayer);
	}
	// Get the correct class sequence as an input
	SetInputMapping(I_Label, *calculator, CCrfCalculationLayer::I_Label);
	
	// Connect the back link with the calculation layer
	backLink->Connect(0, *calculator, CCrfCalculationLayer::O_ClassSeqLogProb);
	calculator->Connect(CCrfCalculationLayer::I_ClassSeqLogProb, *backLink);

	// Connect the outputs of the calculation layer with the main layer outputs
	SetOutputMapping(O_BestPrevClass, *calculator, CCrfCalculationLayer::O_BestPrevClass);
	SetOutputMapping(O_ClassSeqLogProb, *calculator, CCrfCalculationLayer::O_ClassSeqLogProb);
	SetOutputMapping(O_LabelLogProb, *calculator, CCrfCalculationLayer::O_LabelLogProb);
}

void CCrfLayer::SetDropoutRate(float newDropoutRate)
{
	if( ( newDropoutRate > 0 && dropOutLayer == 0 ) || ( newDropoutRate <= 0 && dropOutLayer != 0 ) ) {
		DeleteAllLayersAndBackLinks();
		buildLayer(newDropoutRate);
	} else if(dropOutLayer != 0) {
		dropOutLayer->SetDropoutRate(newDropoutRate);
	}
}

static const int CrfLayerVersion = 2000;

void CCrfLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CrfLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CRecurrentLayer::Serialize( archive );

	if( archive.IsLoading() ) {
		hiddenLayer = CheckCast<CFullyConnectedLayer>( GetLayer( hiddenLayer->GetName() ) );
		if( HasLayer(DropOutName) ) {
			dropOutLayer = CheckCast<CDropoutLayer>(GetLayer(DropOutName));
		}
		calculator = CheckCast<CCrfCalculationLayer>( GetLayer( calculator->GetName() ) );
		backLink = CheckCast<CBackLinkLayer>( GetLayer( backLink->GetName() ) );
	}
}

///////////////////////////////////////////////////////////////////////////////////
// CDnnCrfSinkLayer

void CBestSequenceLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture( GetInputCount() == 2, GetName(), "CRF layer with incorrect numbers of input and output" );
	CheckArchitecture( inputDescs[I_BestPrevClass].HasEqualDimensions(inputDescs[I_ClassSeqLogProb]),
		GetName(), "incorrect inputs size" );
	// Create the blob that will store the best sequence
	outputDescs[0] = CBlobDesc( CT_Int );
	outputDescs[0].SetDimSize( BD_BatchLength, inputDescs[I_BestPrevClass].BatchLength() );
	outputDescs[0].SetDimSize( BD_BatchWidth, inputDescs[I_BestPrevClass].BatchWidth() );
}

void CBestSequenceLayer::BackwardOnce()
{
	inputDiffBlobs[I_BestPrevClass]->Clear();
	inputDiffBlobs[I_ClassSeqLogProb]->Clear();
}

void CBestSequenceLayer::RunOnce()
{
	int batchLength = inputBlobs[I_BestPrevClass]->GetBatchLength();
	int batchWidth = inputBlobs[I_BestPrevClass]->GetBatchWidth();
	int numberOfClasses = inputBlobs[I_BestPrevClass]->GetObjectSize();

	CConstFloatHandle classSeqLogProbData = inputBlobs[I_ClassSeqLogProb]->GetData( {batchLength - 1} );
	CFloatHandleStackVar maxProbabilities(MathEngine(), batchWidth);
	CIntHandleStackVar bestLabelsHandle(MathEngine(), batchWidth);
	// Find the last labels of the best sequences
	MathEngine().FindMaxValueInRows(classSeqLogProbData, batchWidth, numberOfClasses,
		maxProbabilities.GetHandle(), bestLabelsHandle.GetHandle(), batchWidth);
	// Find the best sequences on backward pass over the CRF output blob
	CVariableMatrix<int> bestLabels(batchLength, batchWidth);
	MathEngine().DataExchangeTyped<int>(bestLabels.Column(batchLength - 1), 
		bestLabelsHandle.GetHandle(), batchWidth);

	if( batchLength > 1 ) {
		// Viterbi backward pass using back-pointers stored in I_BestPrevClass
		CVariableMatrix<int> bestPrevLabels(batchLength * batchWidth, numberOfClasses);
		inputBlobs[I_BestPrevClass]->CopyTo(bestPrevLabels.Column(0), bestPrevLabels.DataSize());
		for(int i = batchLength - 1; i >= 1; i--) {
			// Get the best labels from the previous sequence elements
			for(int j = 0; j < batchWidth; j++) {
				bestLabels(i - 1, j) = bestPrevLabels(i * batchWidth + j, bestLabels(i, j));
			}
		}
	}
	outputBlobs[0]->CopyFrom<int>(bestLabels.Column(0));
}

static const int BestSequenceLayerVersion = 2000;

void CBestSequenceLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( BestSequenceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

} // namespace NeoML
