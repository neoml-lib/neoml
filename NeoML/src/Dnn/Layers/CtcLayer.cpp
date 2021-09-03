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

// Based on Alex Graves, "Supervised Sequence Labelling with Recurrent Neural Networks", chapter 7.

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/CtcLayer.h>
#include <float.h>

namespace NeoML {

static const float MaxGradientValue = 1e+6;

////////////////////////////////////////////////////////////////////////////////////////
// CCtcLossLayer

CCtcLossLayer::CCtcLossLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnCtcLossLayer", false ),
	lossWeight( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	loss( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	lossGradientDivider( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	minGradient( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	maxGradient( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	blankLabel( 0 ),
	allowBlankLabelSkip( false )
{
	SetLossWeight(1.);
	loss->GetData().SetValue( 0 );
	minGradient->GetData().SetValue( -MaxGradientValue );
	maxGradient->GetData().SetValue( MaxGradientValue );
}

void CCtcLossLayer::SetMaxGradientValue(float maxValue)
{
	NeoAssert(maxValue > 0);

	minGradient->GetData().SetValue(-maxValue);
	maxGradient->GetData().SetValue(maxValue);
}

void CCtcLossLayer::Reshape()
{
	CheckInputs();

	CheckArchitecture(outputDescs.IsEmpty(), GetName(), "CCtcLossLayer has no output");
	CheckArchitecture(!GetDnn()->IsRecurrentMode(),
		GetName(), "ctc loss layer inside the recurrent composite layer" );
	CheckArchitecture( GetInputCount() >= 2 && GetInputCount() <= 5,
		GetName(), "CCtcLossLayer must have two to five inputs" );

	const CBlobDesc& labels = inputDescs[I_Labels];
	const bool hasLabelsLengths = GetInputCount() > I_LabelsLengths;
	const bool hasInputLengths = GetInputCount() > I_InputLengths;

	const int batchWidth = labels.BatchWidth();
	const int labelsMaxLength = labels.BatchLength();

	CheckArchitecture( inputDescs[I_Result].BatchWidth() == inputDescs[I_Labels].BatchWidth(), 
		GetName(), "loss layer result batch size doesn't match labels batch size" );
	CheckArchitecture( inputDescs[I_Result].ObjectSize() >= blankLabel, GetName(),
		"too small classes count" );
	CheckArchitecture( inputDescs[I_Labels].BatchLength() >= 1 && inputDescs[I_Labels].ObjectSize() == 1, 
		GetName(), "incorrect label size" );
	CheckArchitecture( allowBlankLabelSkip || hasLabelsLengths || labelsMaxLength * 2 + 1 <= inputDescs[I_Result].BatchLength(),
		GetName(), "too small input length" );
	if( hasLabelsLengths ) {
		CheckArchitecture( inputDescs[I_LabelsLengths].BatchLength() == 1 &&
			inputDescs[I_LabelsLengths].BatchWidth() == batchWidth &&
			inputDescs[I_LabelsLengths].ObjectSize() == 1,
			GetName(), "CCtcLossLayer: incorrect labels lengths blob dimensions" );
	}
	if( hasInputLengths ) {
		CheckArchitecture( inputDescs[I_InputLengths].BatchLength() == 1 &&
			inputDescs[I_InputLengths].BatchWidth() == batchWidth &&
			inputDescs[I_InputLengths].ObjectSize() == 1,
			GetName(), "CCtcLossLayer: incorrect inputs lengths blob dimensions" );
	}
	if( GetInputCount() > I_LabelWeights ) {
		CheckArchitecture( inputDescs[I_Result].BatchWidth() == inputDescs[I_LabelWeights].BatchWidth(),
			GetName(), "weights batch size doesn't match result batch size" );
		CheckArchitecture( inputDescs[I_LabelWeights].BatchLength() == 1 && inputDescs[I_LabelWeights].ObjectSize() == 1,
			GetName(), "weight's batchLength and objectSize must have be equal to 1" );
	}

	lossGradient = 0;

	CFloatHandleStackVar tempLossDivider( MathEngine() );
	tempLossDivider.SetValue( 1.f / inputDescs[I_Result].BatchWidth() );
	MathEngine().VectorEltwiseMultiply( tempLossDivider.GetHandle(), lossWeight->GetData(),
		lossGradientDivider->GetData(), 1 );
}

void CCtcLossLayer::RunOnce()
{
	const CBlobDesc& result = inputBlobs[I_Result]->GetDesc();
	const CBlobDesc& label = inputBlobs[I_Labels]->GetDesc();
	if( IsBackwardPerformed() && ( lossGradient == nullptr || !lossGradient->HasEqualDimensions( inputBlobs[I_Result] ) ) ) {
		lossGradient = inputBlobs[I_Result]->GetClone();
	}
	MathEngine().CtcLossForward( result.BatchLength(), result.BatchWidth() * result.ListSize(), result.ObjectSize(),
		label.BatchLength(), blankLabel, allowBlankLabelSkip, inputBlobs[I_Result]->GetData(), inputBlobs[I_Labels]->GetData<int>(),
		inputBlobs.Size() <= I_LabelsLengths ? CConstIntHandle() : inputBlobs[I_LabelsLengths]->GetData<int>(),
		inputBlobs.Size() <= I_InputLengths ? CConstIntHandle() : inputBlobs[I_InputLengths]->GetData<int>(),
		inputBlobs.Size() <= I_LabelWeights ? CConstFloatHandle() : inputBlobs[I_LabelWeights]->GetData(),
		loss->GetData(), IsBackwardPerformed() ? lossGradient->GetData() : CFloatHandle() );
}

void CCtcLossLayer::BackwardOnce()
{
	// Take weights into account
	if( inputBlobs.Size() > I_LabelWeights ) {
		MathEngine().Multiply1DiagMatrixByMatrix( lossGradient->GetBatchLength(), inputBlobs[I_LabelWeights]->GetData(),
			lossGradient->GetBatchWidth(), lossGradient->GetData(), lossGradient->GetObjectSize(),
			inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize() );
		MathEngine().VectorMultiply( inputDiffBlobs[I_Result]->GetData(),
			inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize(),
			lossGradientDivider->GetData() );
	} else {
		MathEngine().VectorMultiply( lossGradient->GetData(), inputDiffBlobs[I_Result]->GetData(),
			inputDiffBlobs[I_Result]->GetDataSize(), lossGradientDivider->GetData() );
	}
	// In case of "huge" gradients the system behavior may be incorrect,
	// so cut these values down
	MathEngine().VectorMinMax(inputDiffBlobs[I_Result]->GetData(), 
		inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize(), 
		minGradient->GetData(), maxGradient->GetData());
}

static const int CtcLossLayerVersion = 2000;

void CCtcLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CtcLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << GetLossWeight();
		archive << maxGradient->GetData().GetValue();
		archive << blankLabel;
		archive << allowBlankLabelSkip;
	} else if( archive.IsLoading() ) {
		float tmp;
		archive >> tmp;
		SetLossWeight(tmp);
		float maxGradientValue = MaxGradientValue;
		archive >> maxGradientValue;
		minGradient->GetData().SetValue( -maxGradientValue );
		maxGradient->GetData().SetValue( maxGradientValue );
		loss->GetData().SetValue( 0 );
		archive >> blankLabel;
		archive >> allowBlankLabelSkip;
		ForceReshape();
	} else {
		NeoAssert( false );
	}
}

CLayerWrapper<CCtcLossLayer> CtcLoss( int blankLabel, bool allowBlankLabelSkip,
	float lossWeight )
{
	return CLayerWrapper<CCtcLossLayer>( "CtcLoss", [=]( CCtcLossLayer* result ) {
		result->SetBlankLabel( blankLabel );
		result->SetAllowBlankLabelSkips( allowBlankLabelSkip );
		result->SetLossWeight( lossWeight );
	} );
}

///////////////////////////////////////////////////////////////////////////////////
// CCtcDecodingLayer

CCtcDecodingLayer::CCtcDecodingLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnCtcDecodingLayer", false ),
	blankLabel(0),
	blankProbabilityThreshold(0.01f),
	arcProbabilityThreshold(0.01f)
{
}

void CCtcDecodingLayer::Reshape()
{
	CheckInputs();
	CheckArchitecture(outputDescs.IsEmpty(), GetName(), "CCtcDecodingLayer has no output");
	CBlobDesc transposedDesc = inputDescs[I_Result];
	transposedDesc.SetDimSize(BD_BatchLength, inputDescs[I_Result].BatchWidth());
	transposedDesc.SetDimSize(BD_BatchWidth, inputDescs[I_Result].BatchLength());
	transposedResult = CDnnBlob::CreateBlob(MathEngine(), CT_Float, transposedDesc);
	resultLogProb = CDnnBlob::CreateWindowBlob(transposedResult, 1);
	bestLabels = CDnnBlob::CreateDataBlob(MathEngine(), CT_Int, transposedResult->GetBatchLength(), 
		transposedResult->GetBatchWidth(), 1);
	lastResults.DeleteAll();
}

void CCtcDecodingLayer::RunOnce()
{
	MathEngine().TransposeMatrix(1, inputBlobs[I_Result]->GetData(), 
		inputBlobs[I_Result]->GetBatchLength(), 1, inputBlobs[I_Result]->GetBatchWidth(),
		inputBlobs[I_Result]->GetObjectSize(), transposedResult->GetData(), 
		transposedResult->GetDataSize());
	// Calculate log(softmax) of the input results
	MathEngine().MatrixSoftmaxByRows(transposedResult->GetData(), 
		transposedResult->GetObjectCount(), transposedResult->GetObjectSize(), 
		transposedResult->GetData());
	MathEngine().VectorLog(transposedResult->GetData(), transposedResult->GetData(), 
		transposedResult->GetDataSize());

	// Find the most probable labels for each coordinate
	CFloatHandleStackVar buffer(MathEngine(), transposedResult->GetObjectCount());
	MathEngine().FindMaxValueInRows(transposedResult->GetData(), 
		transposedResult->GetObjectCount(), transposedResult->GetObjectSize(), buffer, 
		bestLabels->GetData<int>(), bestLabels->GetDataSize());

	inputBlobs.CopyTo(lastResults);
}

void CCtcDecodingLayer::BackwardOnce()
{
}

static void integrateResultMatrix(CVariableMatrix<float>& resultMatrix)
{
	for(int i = 1; i < resultMatrix.SizeX(); i++) {
		float* prevCol = resultMatrix.Column(i - 1);
		float* curCol = resultMatrix.Column(i);
		for(int j = 0; j < resultMatrix.SizeY(); j++) {
			curCol[j] += prevCol[j];
		}
	}
}

bool CCtcDecodingLayer::BuildGLD(int sequenceNumber, CLdGraph<CCtcGLDArc>& gld) const
{
	gld.DeleteAll();
	int sequenceLength = lastResults[I_Result]->GetBatchLength();
	int labelsCount = lastResults[I_Result]->GetChannelsCount();
	// Find the total logarithm of probability of each label 
	// from the start to the current index in the sequence
	resultLogProb->SetParentPos(sequenceNumber);
	CVariableMatrix<float> resultMatrix(sequenceLength, labelsCount);
	resultLogProb->CopyTo( resultMatrix.Column( 0 ) );
	integrateResultMatrix(resultMatrix);
	// Find the best labels for each index
	CArray<int> bestLabelsArray;
	bestLabelsArray.SetSize(sequenceLength);
	MathEngine().DataExchangeTyped( bestLabelsArray.GetPtr(), bestLabels->GetData<const int>( {sequenceNumber} ), sequenceLength );
	// Find the possible start and end of blank label ranges
	float labelLogProbThreshold = logf(arcProbabilityThreshold);
	float blankLogProbThreshold = logf(blankProbabilityThreshold);
	CDynamicBitSet<> blankStarts, blankEnds;
	blankEnds.Set(0);
	blankStarts.Set(sequenceLength);
	int certainBlankStart = NotFound;
	for(int x = 0; x < sequenceLength; x++) {
		if(bestLabelsArray[x] == blankLabel) {
			if(certainBlankStart == NotFound) {
				// The start of highly probable blank range
				blankStarts.Set(x);
				certainBlankStart = x;
			}
		} else {
			if(certainBlankStart != NotFound) {
				// The end of highly probable blank range
				blankEnds.Set(x);
				float blankLogProb = resultMatrix(x - 1, blankLabel); 
				if(certainBlankStart > 0) {
					blankLogProb -= resultMatrix(certainBlankStart - 1, blankLabel);
				}
				auto arc = FINE_DEBUG_NEW CCtcGLDArc(certainBlankStart, x);
				arc->Label = blankLabel;
				arc->LogProb = blankLogProb;
				gld.InsertArc(arc);
			} else {
				// Out of the probable blank range
				float blankLogProb = resultMatrix(x, blankLabel);
				if(x > 0) {
					blankLogProb -= resultMatrix(x - 1, blankLabel);
				}
				if(blankLogProb >= blankLogProbThreshold) {
					blankStarts.Set(x);
					blankEnds.Set(x + 1);
					auto arc = FINE_DEBUG_NEW CCtcGLDArc(x, x + 1);
					arc->Label = blankLabel;
					arc->LogProb = blankLogProb;
					gld.InsertArc(arc);
				}
			}
			certainBlankStart = NotFound;
		}
	}
	// Build an arc for the final highly probable blank range
	if(certainBlankStart != NotFound) {
		float blankLogProb = resultMatrix(sequenceLength - 1, blankLabel); 
		if(certainBlankStart > 0) {
			blankLogProb -= resultMatrix(certainBlankStart - 1, blankLabel);
		}
		auto arc = FINE_DEBUG_NEW CCtcGLDArc(certainBlankStart, sequenceLength);
		arc->Label = blankLabel;
		arc->LogProb = blankLogProb;
		gld.InsertArc(arc);
	}
	// Build a linear division graph
	for(int x = 0; x < sequenceLength; x++) {
		if(!blankEnds.Has(x)) {
			continue;
		}
		for(int y = x + 1; y <= sequenceLength; y++ ) {
			if(!blankStarts.Has(y)) {
				continue;
			}
			for(int l = 0; l < labelsCount; l++) {
				if(l == blankLabel) {
					continue;
				}
				float labelLogProb = resultMatrix(y - 1, l);
				if(x > 0) {
					labelLogProb -= resultMatrix(x - 1, l);
				}
				if(labelLogProb >= labelLogProbThreshold) {
					auto arc = FINE_DEBUG_NEW CCtcGLDArc(x, y);
					arc->Label = l;
					arc->LogProb = labelLogProb;
					gld.InsertArc(arc);
				}
			}
		}
	}
	gld.CalculateBestPathQuality(-FLT_MAX / 2);
	return gld.VerifyPath(0, sequenceLength);
}

void CCtcDecodingLayer::GetBestSequence(int sequenceNumber, CArray<int>& bestLabelSequence) const
{
	bestLabelSequence.DeleteAll();

	if(lastResults.IsEmpty()) {
		return;
	}
	// Find the best label for each index in the sequence
	int sequenceLength = lastResults[I_Result]->GetBatchLength();
	if( lastResults.Size() > I_InputLengths ) {
		CArray<int> lengths;
		lengths.SetSize( lastResults[I_InputLengths]->GetDataSize() );
		lastResults[I_InputLengths]->CopyTo( lengths.GetPtr() );
		sequenceLength = min( lengths[sequenceNumber], sequenceLength );
	}
	CArray<int> bestLabelsArray;
	bestLabelsArray.SetSize(sequenceLength);
	MathEngine().DataExchangeTyped(bestLabelsArray.GetPtr(), bestLabels->GetData<const int>( {sequenceNumber} ), sequenceLength);
	
	for(int i = 0; i < bestLabelsArray.Size(); i++) {
		int l = bestLabelsArray[i];
		if(l != blankLabel && 
			(i == 0 || l != bestLabelsArray[i - 1]) ) 
		{
			bestLabelSequence.Add(l);
		}
	}
}

static const int CtcDecodingLayerVersion = 2000;

void CCtcDecodingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CtcDecodingLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << blankLabel;
		archive << blankProbabilityThreshold;
		archive << arcProbabilityThreshold;
	} else if( archive.IsLoading() ) {
		archive >> blankLabel;
		archive >> blankProbabilityThreshold;
		archive >> arcProbabilityThreshold;
		ForceReshape();
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
