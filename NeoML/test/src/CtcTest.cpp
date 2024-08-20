/* Copyright © 2021 ABBYY Production LLC

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

#include <memory>
#include <cmath>

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

namespace NeoMLTest {

// The layer that implements connectionist temporal classification (CTC) 
// for training recurrent networks to recognize sequences
class CCtcLossNaiveLayer : public CBaseLayer {
public:
	explicit CCtcLossNaiveLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& /* archive */ ) override { NeoAssert( false ); }

	// The layer may have 2 to 5 inputs
	enum TInput {
		// The activations of the previous layer, of the dimensions:
		//  (BatchLength=the counter) * (BatchWidth=batch size) * (the number of classes)
		I_Result = 0,

		// The class labels, a blob with integer data, of the dimensions:
		//  (BatchLength=the maximum label length) * (BatchWidth=batch size) * 1
		I_Labels = 1,

		// The class labels length (optional), a blob with integer data, of the dimensions:
		//  (BatchLength=1) * (BatchWidth=batch size) * 1
		I_LabelsLengths = 2,

		// The input sequence length (optional), a blob with integer data, of the dimensions:
		//  (BatchLength=1) * (BatchWidth=batch size) * 1
		I_InputLengths = 3,

		// Training sample weights (optional), of the dimensions:
		//  (BatchLength=1) * (BatchWidth=batch size)* 1
		I_LabelWeights = 4
	};

	// The blank label, to be used as a space between other labels:
	int GetBlankLabel() const { return blankLabel; }
	void SetBlankLabel(int _blankLabel) { blankLabel = _blankLabel; }

	// Total loss weight
	float GetLossWeight() const { return lossWeight->GetData().GetValue(); }
	void SetLossWeight(float _lossWeight) { lossWeight->GetData().SetValue(_lossWeight); }

	// Gets the last loss value
	float GetLastLoss() const { return loss->GetData().GetValue(); }

	// The maximum loss gradient value
	// The system may not function as intended with very large loss gradient,
	// so we don't recommend changing this value
	float GetMaxGradientValue() const { return maxGradient->GetData().GetValue(); }
	void SetMaxGradientValue(float maxValue);

	// Indicates if the blank labels may be skipped when aligning
	bool GetAllowBlankLabelSkips() { return allowBlankLabelSkip; }
	void SetAllowBlankLabelSkips( bool enabled ) { allowBlankLabelSkip = enabled; }

	const CPtr<CDnnBlob> GetLastGradient() const { return lossGradient; }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	int BlobsForBackward() const override { return 0; }

private:
	CPtr<CDnnBlob> lossWeight; // scale multiplier for the loss function
	CPtr<CDnnBlob> loss; // the loss value on the last tep
	CPtr<CDnnBlob> lossDivider; // the averaging factor for calculating the loss value
	CPtr<CDnnBlob> lossGradientDivider; // the averaging factor for calculating the loss gradient (taking lossWeight into account)
	CPtr<CDnnBlob> weights;	// the vector weights

	CPtr<CDnnBlob> minGradient;
	CPtr<CDnnBlob> maxGradient;

	int blankLabel; // the blank label
	CPtr<CDnnBlob> paddedLabels; // the sequence of labels separated by blanks { 2 * LabelLength + 1, BW, 1 } int
	CPtr<CDnnBlob> nonBlanksMask; // 0.0 for the blanks, 1.0 for the rest { 2 * LabelLength + 1 }
	CPtr<CDnnBlob> labelRows; // the matrix row numbers for filling in paddedLabels
	CPtr<CDnnBlob> logAlpha, logBeta; // the blobs with logarithms of prefix and suffix sequence probabilities { InputLength, 2 * LabelLength + 1, BW }
	CPtr<CDnnBlob> blankSkipMask; // the blobs with masks for elements with blanks removed { 2 * LabelLength + 1, BW }
	CPtr<CDnnBlob> logBetaPrev2; // the blob with logarithms of suffixes shifted 2 steps back in time { 2 * LabelLength + 1, BW }
	CPtr<CDnnBlob> resultProb; // softmax(I_Result)	{ InputLength, BW, Classes }
	CPtr<CDnnBlob> resultLogProb; // log(softmax(I_Result)) { InputLength, BW, Classes }
	CPtr<CDnnBlob> paddingResultValue; // contains the padding for the result { Classes }

	CPtr<CDnnBlob> resultProbWindow;
	CPtr<CDnnBlob> resultLogProbWindow;
	CPtr<CDnnBlob> logAlphaWindow;
	CPtr<CDnnBlob> logAlphaPrevWindow;
	CPtr<CDnnBlob> logBetaWindow;
	// total logarithm of the probability of sequences with a given alignment at a given moment
	CPtr<CDnnBlob> logAlphaBeta; // { 2 * LabelLength + 1, BW }
	
	CPtr<CDnnBlob> lossGradient; // loss function gradient { InputLength, BW, Classes }
	CPtr<CDnnBlob> lossGradientWindow;
	CPtr<CDnnBlob> probSum;     // a temporary blob for calculating gradient
	CPtr<CDnnBlob> rowIndices;  // the row indices blob

	// The arrays of indices for initializing beta on a backward pass
	// They point at where the labels end in the length and position dimensions in a batch
	CPtr<CDnnBlob> endOfLabelPosition;
	CPtr<CDnnBlob> endOfLabelSample;

	// The integer constant -1
	CPtr<CDnnBlob> minusOneInt;
	// The array of float zeros of BatchWidth size
	CPtr<CDnnBlob> batchOfZeros;

	bool allowBlankLabelSkip; // indicates if blanks between different labels may be skipped

	void calculateForwardVariables();
	void calculateBackwardVariables( CDnnBlob* labelsLength, CDnnBlob* inputsLengths );
	void calculateGradient(CFloatHandle totalLogProb);
	void calculateBlankSkipMasks();
	void applyInputLengthsPadding( CDnnBlob* inputLengths, CDnnBlob* paddingBlob,
		CDnnBlob* targetBlob, CDnnBlob* targetBlobWindow );
};

static const float MaxGradientValue = 1e+6;
static const float logZero = -FLT_MAX / 4;
// log(1-). Using 0.0 may lead to denormalized numbers
static const float logOneNeg = -FLT_MIN * 2;

/////////////////////////////////////////////////////////////////////////////////////////
// These two methods are only called from Reshape, so their efficiency is not critical
// May be rewritten as CMathEngine primitives if needed

// Fills a matrix with the two-dimensional arithmetic progression values:
//  result[i, j] = start + i * rowStep + j * colStep
static void matrixFillLinear( IMathEngine& mathEngine,
	const CIntHandle& resultHandle, int height, int width,
	int start, int rowStep, int colStep )
{
	CArray<int> result;
	result.SetSize( width * height );

	for( int i = 0; i < height; i++ ) {
		for( int j = 0; j < width; j++ ) {
			result[i * width + j] = start + i * rowStep + j * colStep;
		}
	}

	mathEngine.DataExchangeTyped( resultHandle, result.GetPtr(), result.Size() );
}

// Fills an array with the arithmetic progression values:
//  result[i] = start + i * step
static void vectorFillLinear( IMathEngine& mathEngine,
	const CIntHandle& resultHandle, int vectorSize, int start, int step )
{
	matrixFillLinear( mathEngine, resultHandle, 1, vectorSize, start, 0, step );
}

// Naive implementation of MatrixLogSumExpByColumns
static void matrixLogSumExpByColumns(CDnnBlob& matrixBlob, int height, int width, const CFloatHandle& resultHandle)
{
	float* matrix = matrixBlob.GetBuffer<float>( 0, matrixBlob.GetDataSize(), true );
	CArray<float> result;
	result.SetSize( width );
	for( int w = 0; w < width; ++w ) {
		float maxVal = matrix[w];
		for( int h = 0; h < height; ++h ) {
			maxVal = fmaxf( maxVal, matrix[h * width + w] );
		}
		float row = 0.f;
		for( int h = 0; h < height; ++h ) {
			row += expf( matrix[h * width + w] - maxVal );
		}
		result[w] = maxVal + logf( row );
	}
	matrixBlob.ReleaseBuffer( matrix, false );

	IMathEngine& mathEngine = *resultHandle.GetMathEngine();
	mathEngine.DataExchangeTyped<float>( resultHandle, result.GetPtr(), width );
}

// Naive implementation of VectorEltwiseLogSumExp
static void vectorEltwiseLogSumExp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	IMathEngine& mathEngine = *firstHandle.GetMathEngine();

	CArray<float> first;
	first.SetSize( vectorSize );
	mathEngine.DataExchangeTyped<float>( first.GetPtr(), firstHandle, vectorSize );

	CArray<float> second;
	second.SetSize( vectorSize );
	mathEngine.DataExchangeTyped<float>( second.GetPtr(), secondHandle, vectorSize );

	CArray<float> result;
	result.SetBufferSize( vectorSize );
	for( int i = 0; i < vectorSize; ++i ) {
		result.Add( first[i] > second[i]
			? first[i] + ::log1pf( ::expf( second[i] - first[i] ) )
			: second[i] + ::log1pf( ::expf( first[i] - second[i] ) ) );
	}
	mathEngine.DataExchangeTyped<float>( resultHandle, result.GetPtr(), vectorSize );
}

static void setVectorToMatrixElements( const CFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CConstFloatHandle& vectorHandle, int vectorSize )
{
	IMathEngine& mathEngine = *matrixHandle.GetMathEngine();

	CArray<float> matrix;
	matrix.SetSize( height * width );
	mathEngine.DataExchangeTyped<float>( matrix.GetPtr(), matrixHandle, height * width );

	CArray<int> rowIndices;
	rowIndices.SetSize( vectorSize );
	mathEngine.DataExchangeTyped<int>( rowIndices.GetPtr(), rowIndicesHandle, vectorSize );

	CArray<int> columnIndices;
	columnIndices.SetSize( vectorSize );
	mathEngine.DataExchangeTyped<int>( columnIndices.GetPtr(), columnIndicesHandle, vectorSize );

	CArray<float> vector;
	vector.SetSize( vectorSize );
	mathEngine.DataExchangeTyped<float>( vector.GetPtr(), vectorHandle, vectorSize );

	for( int i = 0; i < vectorSize; i++ ) {
		matrix[rowIndices[i] * width + columnIndices[i]] = vector[i];
	}

	mathEngine.DataExchangeTyped<float>( matrixHandle, matrix.GetPtr(), height * width );
}

// LogSumExp for two inputs
static inline float LogSumExpFunc(float f, float s)
{
	if(f >= s) {
		return f + log1pf(expf(s - f));
	} else {
		return s + log1pf(expf(f - s));
	}
}

static void eltwiseLogSumExpVectorToMatrixElements(const CFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CConstFloatHandle& vectorHandle, int vectorSize)
{
	IMathEngine& mathEngine = *matrixHandle.GetMathEngine();

	CArray<float> matrix;
	matrix.SetSize( height * width );
	mathEngine.DataExchangeTyped<float>( matrix.GetPtr(), matrixHandle, height * width );

	CArray<int> rowIndices;
	rowIndices.SetSize( vectorSize );
	mathEngine.DataExchangeTyped<int>( rowIndices.GetPtr(), rowIndicesHandle, vectorSize );

	CArray<int> columnIndices;
	columnIndices.SetSize( vectorSize );
	mathEngine.DataExchangeTyped<int>( columnIndices.GetPtr(), columnIndicesHandle, vectorSize );

	CArray<float> vector;
	vector.SetSize( vectorSize );
	mathEngine.DataExchangeTyped<float>( vector.GetPtr(), vectorHandle, vectorSize );

	for(int i = 0; i < vectorSize; i++) {
		const int rowIndex = rowIndices[i];
		const int columnIndex = columnIndices[i];
		if(rowIndex >= 0 && rowIndex < height &&
			columnIndex >= 0 && columnIndex < width) {
			const int matrixIndex = rowIndex * width + columnIndex;
			matrix[matrixIndex] = LogSumExpFunc(vector[i], matrix[matrixIndex]);
		}
	}

	mathEngine.DataExchangeTyped<float>( matrixHandle, matrix.GetPtr(), height * width );
}

/////////////////////////////////////////////////////////////////////////////////////////
// CCtcLossNaiveLayer

CCtcLossNaiveLayer::CCtcLossNaiveLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnCtcLossLayer", false ),
	lossWeight( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	loss( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	lossDivider( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	lossGradientDivider( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	minGradient( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	maxGradient( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	blankLabel( 0 ),
	minusOneInt( CDnnBlob::CreateVector( mathEngine, CT_Int, 1 ) ),
	allowBlankLabelSkip( false )
{
	SetLossWeight(1.);
	loss->GetData().SetValue( 0 );
	minGradient->GetData().SetValue( -MaxGradientValue );
	maxGradient->GetData().SetValue( MaxGradientValue );
	minusOneInt->GetData<int>().SetValue( -1 );
}

void CCtcLossNaiveLayer::SetMaxGradientValue(float maxValue)
{
	NeoAssert(maxValue > 0);

	minGradient->GetData().SetValue(-maxValue);
	maxGradient->GetData().SetValue(maxValue);
}

void CCtcLossNaiveLayer::Reshape()
{
	CheckInputs();

	CheckArchitecture(outputDescs.IsEmpty(), GetPath(), "CCtcLossNaiveLayer has no output");
	CheckArchitecture(!GetDnn()->IsRecurrentMode(),
		GetPath(), "ctc loss layer inside the recurrent composite layer" );
	CheckArchitecture( GetInputCount() >= 2 && GetInputCount() <= 5,
		GetPath(), "CCtcLossNaiveLayer must have two to five inputs" );

	const CBlobDesc& labels = inputDescs[I_Labels];
	const bool hasLabelsLengths = GetInputCount() > I_LabelsLengths;
	const bool hasInputLengths = GetInputCount() > I_InputLengths;

	const int batchWidth = labels.BatchWidth();
	const int labelsMaxLength = labels.BatchLength();

	CheckArchitecture( inputDescs[I_Result].BatchWidth() == inputDescs[I_Labels].BatchWidth(), 
		GetPath(), "loss layer result batch size doesn't match labels batch size" );
	CheckArchitecture( inputDescs[I_Result].ObjectSize() >= blankLabel, GetPath(),
		"too small classes count" );
	CheckArchitecture( inputDescs[I_Labels].BatchLength() >= 1 && inputDescs[I_Labels].ObjectSize() == 1, 
		GetPath(), "incorrect label size" );
	CheckArchitecture( allowBlankLabelSkip || hasLabelsLengths || labelsMaxLength * 2 + 1 <= inputDescs[I_Result].BatchLength(),
		GetPath(), "too small input length" );
	if( hasLabelsLengths ) {
		CheckArchitecture( inputDescs[I_LabelsLengths].BatchLength() == 1 &&
			inputDescs[I_LabelsLengths].BatchWidth() == batchWidth &&
			inputDescs[I_LabelsLengths].ObjectSize() == 1,
			GetPath(), "CCtcLossNaiveLayer: incorrect labels lengths blob dimensions" );
	}
	if( hasInputLengths ) {
		CheckArchitecture( inputDescs[I_InputLengths].BatchLength() == 1 &&
			inputDescs[I_InputLengths].BatchWidth() == batchWidth &&
			inputDescs[I_InputLengths].ObjectSize() == 1,
			GetPath(), "CCtcLossNaiveLayer: incorrect inputs lengths blob dimensions" );
	}
	if( GetInputCount() > I_LabelWeights ) {
		CheckArchitecture( inputDescs[I_Result].BatchWidth() == inputDescs[I_LabelWeights].BatchWidth(),
			GetPath(), "weights batch size doesn't match result batch size" );
		CheckArchitecture( inputDescs[I_LabelWeights].BatchLength() == 1 && inputDescs[I_LabelWeights].ObjectSize() == 1,
			GetPath(), "weight's batchLength and objectSize must have be equal to 1" );
	} else {
		weights = CDnnBlob::CreateVector( MathEngine(), CT_Float, inputDescs[I_Result].BatchWidth() );
		weights->Fill( 1.f );
	}

	// Create a blob with label sequences separated by spaces
	const int paddedLabelsMaxLength = labelsMaxLength * 2 + 1;
	paddedLabels = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, paddedLabelsMaxLength, batchWidth, 1 );

	// The table for recalculating label index when inserting spaces
	labelRows = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, labelsMaxLength, 1, 1 );
	vectorFillLinear( MathEngine(), labelRows->GetData<int>(), labelsMaxLength, 1, 2 );

	if( allowBlankLabelSkip ) {
		nonBlanksMask = CDnnBlob::CreateVector( MathEngine(), CT_Float, inputDescs[I_Labels].BatchLength() * 2 + 1 );
		nonBlanksMask->Fill( 0.0f );
		CFloatHandleStackVar fillValuesVariable(MathEngine(), 2);
		const float fillValues[] = { 0.0f, 1.0f };
		MathEngine().DataExchangeTyped( fillValuesVariable.GetHandle(), fillValues, 2 );
		MathEngine().AddVectorToMatrixRows(1, nonBlanksMask->GetData(), nonBlanksMask->GetData(),
			inputDescs[I_Labels].BatchLength(), 2, fillValuesVariable);

		blankSkipMask = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, paddedLabels->GetBatchLength(), 
			inputDescs[I_Result].BatchWidth() );
	}

	// A table for the indices in the input activation matrix
	rowIndices = paddedLabels->GetClone();
	matrixFillLinear( MathEngine(),
		rowIndices->GetData<int>(), paddedLabelsMaxLength, batchWidth, 0, 0, 1 );

	// Index arrays for initializing beta on the backward pass
	if( hasLabelsLengths ) {
		endOfLabelPosition = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, 1, batchWidth, 1 );
		endOfLabelSample = endOfLabelPosition->GetClone();
		vectorFillLinear( MathEngine(), endOfLabelSample->GetData<int>(), batchWidth, 0, 1 );
		batchOfZeros = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 1, batchWidth, 1 );
		batchOfZeros->Fill( 0.f );
	}

	// Create blobs for logarithms of probability of prefixes logAlpha and suffixes logBeta
	logAlpha = CDnnBlob::CreateDataBlob(MathEngine(), CT_Float, inputDescs[I_Result].BatchLength(), 
		paddedLabels->GetBatchLength(), inputDescs[I_Result].BatchWidth());
	logBeta = CDnnBlob::CreateDataBlob(MathEngine(), CT_Float, inputDescs[I_Result].BatchLength(), 
		paddedLabels->GetBatchLength(), inputDescs[I_Result].BatchWidth());

	// The padding blob
	const int classesCount = inputDescs[I_Result].ObjectSize();
	paddingResultValue = CDnnBlob::CreateVector( MathEngine(), CT_Float, classesCount );

	resultProb = CDnnBlob::CreateBlob( MathEngine(), inputDescs[I_Result] );
	resultLogProb = resultProb->GetClone();

	resultProbWindow = CDnnBlob::CreateWindowBlob(resultProb, 1);
	probSum = resultProbWindow->GetClone();
	resultLogProbWindow = CDnnBlob::CreateWindowBlob(resultLogProb, 1);
	logAlphaWindow = CDnnBlob::CreateWindowBlob(logAlpha, 1);
	logAlphaPrevWindow = CDnnBlob::CreateWindowBlob(logAlpha, 1);
	logBetaWindow = CDnnBlob::CreateWindowBlob(logBeta, 1);
	logAlphaBeta = logBetaWindow->GetClone();
	logBetaPrev2 = logBetaWindow->GetClone();
	lossGradient = 0;
	lossGradientWindow = 0;

	CFloatHandleStackVar tempLossDivider( MathEngine() );
	tempLossDivider.SetValue( 1.f / inputDescs[I_Result].BatchWidth() );
	MathEngine().VectorEltwiseMultiply( tempLossDivider.GetHandle(), lossWeight->GetData(),
		lossGradientDivider->GetData(), 1 );
	// Change the sign before the lossDivider:
	MathEngine().VectorNeg( tempLossDivider.GetHandle(), lossDivider->GetData(), 1 );
}

// Calculates the logarithms of prefix probability logAlpha on a forward pass
void CCtcLossNaiveLayer::calculateForwardVariables()
{
	NeoAssert(paddedLabels->GetBatchLength() == logAlpha->GetBatchWidth());

	const int T = logAlpha->GetBatchLength();
	const int U = logAlpha->GetBatchWidth();
	// Initialize
	resultLogProbWindow->SetParentPos(0);
	logAlphaWindow->SetParentPos(0);
	// The sequence may start with a space or with the first element
	MathEngine().VectorFill(logAlphaWindow->GetObjectData( 0 ), 0.f, 
		logAlphaWindow->GetObjectSize() * 2);
	MathEngine().VectorFill(
		logAlphaWindow->GetObjectData( 2 ), logZero,
		logAlphaWindow->GetObjectSize() * (U - 2) );
	// Add the logarithm of probability of label recognition
	MathEngine().AddMatrixElementsToVector(resultLogProbWindow->GetData(), 
		resultLogProbWindow->GetBatchWidth(), resultLogProbWindow->GetObjectSize(),
		rowIndices->GetData<int>(), paddedLabels->GetData<int>(),
		logAlphaWindow->GetData(), logAlphaWindow->GetDataSize());

	const CPtr<CDnnBlob>& logAlphaShift2Buffer = logAlphaBeta;

	// Align the result sequence T elements long with the labels and spaces sequence U elements long
	for(int t = 1; t < T; t++) {
		resultLogProbWindow->SetParentPos(t);
		logAlphaWindow->SetParentPos(t);
		logAlphaPrevWindow->SetParentPos(t - 1);
		// Add up the alternative pairings after the previous moment in time
		MathEngine().VectorCopy( logAlphaWindow->GetObjectData( 0 ),
			logAlphaPrevWindow->GetObjectData( 0 ), logAlphaWindow->GetObjectSize() );
		vectorEltwiseLogSumExp( logAlphaPrevWindow->GetObjectData( 0 ),
			logAlphaPrevWindow->GetObjectData( 1 ), logAlphaWindow->GetObjectData( 1 ),
			logAlphaWindow->GetObjectSize() * (U - 1) );
		if( allowBlankLabelSkip ) {
			// If label[i] != blank and label[i] != label[i-2], the labels may be put together with no space:
			// label[i-2];label[i] -> label[i-2];blank;label[i]
			MathEngine().VectorAdd( logAlphaPrevWindow->GetObjectData( 0 ),
				blankSkipMask->GetObjectData( 0 ), logAlphaShift2Buffer->GetData(),
				logAlphaWindow->GetObjectSize() * (U - 2) );
			vectorEltwiseLogSumExp( logAlphaWindow->GetObjectData( 2 ),
				logAlphaShift2Buffer->GetData(), logAlphaWindow->GetObjectData( 2 ),
				logAlphaWindow->GetObjectSize() * (U - 2) );
		}
		// Add the logarithm of probability of label recognition
		MathEngine().AddMatrixElementsToVector(resultLogProbWindow->GetData(), 
			resultLogProbWindow->GetBatchWidth(), resultLogProbWindow->GetObjectSize(),
			rowIndices->GetData<int>(), paddedLabels->GetData<int>(), 
			logAlphaWindow->GetData(), logAlphaWindow->GetDataSize());
	}
}

// Calculates the logarithms of suffixes probability logBeta on a backward pass
// The difference in calculating logAlpha[t] and logBeta[t] stems from the fact
// that logAlpha[t] + logBeta[t] must be equal to the logarithm of probability of recognizing the sequence
void CCtcLossNaiveLayer::calculateBackwardVariables( CDnnBlob* labelsLengths, CDnnBlob* inputsLengths )
{
	NeoAssert(paddedLabels->GetBatchLength() == logBeta->GetBatchWidth());

	CFloatHandleStackVar logZeroVal( MathEngine() );
	logZeroVal.SetValue( logZero );
	CFloatHandleStackVar logOneNegVal( MathEngine() );
	logOneNegVal.SetValue( logOneNeg );
	
	const int T = logBeta->GetBatchLength();
	const int U = logBeta->GetBatchWidth();
	const int batchWidth = logBeta->GetObjectSize();
	// Initialize
	resultLogProbWindow->SetParentPos(T - 1);
	logBetaWindow->SetParentPos(T - 1);
	// Use the logAlphaBeta and nonBlanksMask arrays as temporary buffers
	const CPtr<CDnnBlob>& logBetaWindowBuffer = logAlphaBeta;

	// The sequence may end either with a space or with the actual last element
	// Therefore logBetaWindow is filled with logZero everywhere except two positions per sample
	// which should be filled with zero
	MathEngine().VectorFill(
		logBetaWindow->GetData(), logZero, logBetaWindow->GetDataSize() );
	if( labelsLengths == 0 ) {
		// Fixed length. Fill the two last positions with zero
		MathEngine().VectorFill(
			logBetaWindow->GetObjectData( U - 2 ), 0.f, 2 * batchWidth );
	} else {
		// Variable length. Fill the (2 * labelsLengths[j]) and (2 * labelsLengths[j] - 1) positions with zero
		MathEngine().VectorAdd(
			labelsLengths->GetData<int>(), labelsLengths->GetData<int>(),
			endOfLabelPosition->GetData<int>(), batchWidth );
		setVectorToMatrixElements(
			logBetaWindow->GetData(), U, batchWidth,
			endOfLabelPosition->GetData<int>(), endOfLabelSample->GetData<int>(),
			batchOfZeros->GetData(), batchWidth );
		MathEngine().VectorAddValue( endOfLabelPosition->GetData<int>(), endOfLabelPosition->GetData<int>(),
			batchWidth, minusOneInt->GetData<int>() );
		if( inputsLengths != 0 ) {
			CArray<int> buffer;
			buffer.SetSize( batchWidth );
			CPtr<CDnnBlob> addition = inputsLengths->GetCopy();
			inputsLengths->CopyTo( buffer.GetPtr() );
			for( int i = 0; i < buffer.Size(); ++i ) {
				buffer[i] = ( buffer[i] < T ) ? 1 : 0;
			}
			MathEngine().DataExchangeTyped( addition->GetData<int>(), buffer.GetPtr(), batchWidth );
			MathEngine().VectorAdd( endOfLabelPosition->GetData<int>(), addition->GetData<int>(), 
				endOfLabelPosition->GetData<int>(), batchWidth );
		}
		setVectorToMatrixElements(
			logBetaWindow->GetData(), U, batchWidth,
			endOfLabelPosition->GetData<int>(), endOfLabelSample->GetData<int>(),
			batchOfZeros->GetData(), batchWidth );
	}

	for(int t = T - 2; t >= 0; t--) {
		resultLogProbWindow->SetParentPos(t + 1);
		logBetaWindow->SetParentPos(t + 1);
		logBetaWindowBuffer->CopyFrom(logBetaWindow);
		logBetaWindow->SetParentPos(t);
		// Add the logarithm of probability of label recognition
		MathEngine().AddMatrixElementsToVector(resultLogProbWindow->GetData(), 
			resultLogProbWindow->GetBatchWidth(), resultLogProbWindow->GetObjectSize(),
			rowIndices->GetData<int>(), paddedLabels->GetData<int>(), 
			logBetaWindowBuffer->GetData(), logBetaWindowBuffer->GetDataSize());
		// Add up the alternative pairings after the previous moment in time
		vectorEltwiseLogSumExp(logBetaWindowBuffer->GetObjectData( 0 ),
			logBetaWindowBuffer->GetObjectData( 1 ), logBetaWindow->GetObjectData( 0 ),
			logBetaWindow->GetObjectSize() * (U - 1));
		if( allowBlankLabelSkip ) {
			// If label[i] != blank and label[i] != label[i+2], the labels may be put together with no space:
			// label[i];label[i+2] -> label[i];blank;label[i+2]
			const CPtr<CDnnBlob>& logBetaShift2Buffer = logBetaPrev2;
			MathEngine().VectorAdd( logBetaWindowBuffer->GetObjectData( 2 ),
				blankSkipMask->GetObjectData( 0 ), logBetaShift2Buffer->GetData(),
				logBetaWindow->GetObjectSize() * (U - 2) );
			vectorEltwiseLogSumExp( logBetaWindow->GetObjectData( 0 ),
				logBetaShift2Buffer->GetData(), logBetaWindow->GetObjectData( 0 ),
				logBetaWindow->GetObjectSize() * (U - 2) );
		}
		MathEngine().VectorCopy( logBetaWindow->GetObjectData( U - 1 ),
			logBetaWindowBuffer->GetObjectData( U - 1 ), logBetaWindow->GetObjectSize() );
	}
}

// Calculates the loss function gradient
void CCtcLossNaiveLayer::calculateGradient(CFloatHandle totalLogProb)
{
	if(lossGradient == 0) {
		lossGradient = inputBlobs[I_Result]->GetClone();
		lossGradientWindow = CDnnBlob::CreateWindowBlob(lossGradient, 1);
	}
	const int T = logAlpha->GetBatchLength(); // the number of "time moments"

// See Alex Graves "Supervised Sequence Labelling with Recurrent Neural Networks", 
// chapter 7.4.1
	for(int t = 0; t < T; t++) {
		probSum->Fill(logZero);
		resultProbWindow->SetParentPos(t);
		logAlphaWindow->SetParentPos(t);
		logBetaWindow->SetParentPos(t);
		lossGradientWindow->SetParentPos(t);
		MathEngine().VectorAdd(logAlphaWindow->GetData(), logBetaWindow->GetData(),
			logAlphaBeta->GetData(), logAlphaBeta->GetDataSize());

		if( allowBlankLabelSkip ) {
			// For the sake of computational stability
			matrixLogSumExpByColumns( *logAlphaBeta, logAlphaBeta->GetBatchWidth(), logAlphaBeta->GetObjectSize(), totalLogProb );
		}

		eltwiseLogSumExpVectorToMatrixElements(probSum->GetData(),
			probSum->GetBatchWidth(), probSum->GetObjectSize(),
			rowIndices->GetData<int>(), paddedLabels->GetData<int>(), 
			logAlphaBeta->GetData(), logAlphaBeta->GetDataSize());

		MathEngine().SubVectorFromMatrixColumns(probSum->GetData(), probSum->GetData(),
			probSum->GetBatchWidth(), probSum->GetObjectSize(), totalLogProb);
		MathEngine().VectorExp(probSum->GetData(), probSum->GetData(), 
			probSum->GetDataSize());

		MathEngine().VectorSub(resultProbWindow->GetData(), probSum->GetData(),
			lossGradientWindow->GetData(), lossGradientWindow->GetDataSize());
	}
	// For the elements after the input sequence end the gradient is 0
	if( inputBlobs.Size() > I_InputLengths ) {
		paddingResultValue->Fill( 0.0f );
		applyInputLengthsPadding( inputBlobs[I_InputLengths], paddingResultValue, lossGradient, lossGradientWindow );
	}
}

// Calculates the space-skipping mask. Formula: mask[i] = log(I{i < len(paddedLabel) - 2} * I{paddedLabel[i] != paddedLabel[i+2]})
void CCtcLossNaiveLayer::calculateBlankSkipMasks()
{
	CFloatHandleStackVar logZeroVal( MathEngine() );
	MathEngine().DataExchangeTyped<float>( logZeroVal, &logZero, 1 );
	// The two last elements are equal to logZero
	MathEngine().VectorFill( blankSkipMask->GetObjectData( paddedLabels->GetBatchLength() - 2 ), 1.0f, 
		blankSkipMask->GetObjectSize() * 2 );
	const int effectiveMaskSize = (paddedLabels->GetBatchLength() - 2) * blankSkipMask->GetObjectSize();
	MathEngine().VectorEqual( paddedLabels->GetData<int>(),
		paddedLabels->GetObjectData<int>( 2 * paddedLabels->GetBatchWidth() ), blankSkipMask->GetData(),
		effectiveMaskSize  );
	MathEngine().VectorMultiply( blankSkipMask->GetData(), blankSkipMask->GetData(),
		blankSkipMask->GetDataSize(), logZeroVal );
}

// Fills the sequence ends (the elements after inputLengths[i]) in targetBlob with the paddingBlob values
// [[t_11, t_12, t_13, t_14, t_15], [t_21, t_22, t_23, t_24, t_25]]; [i_1 = 3, i_2 = 4]; pad 
//		=> [[t_11, t_12, t_13, pad, pad], [t_21, t_22, t_23, t_24, pad]]
void CCtcLossNaiveLayer::applyInputLengthsPadding( CDnnBlob* inputLengths, CDnnBlob* paddingBlob, 
	CDnnBlob* targetBlob, CDnnBlob* targetBlobWindow )
{
	NeoAssert( paddingBlob->GetDataSize() == targetBlobWindow->GetObjectSize() );
	if( inputLengths == 0 ) {
		return;
	}
	const int maxInputLength = targetBlob->GetBatchLength();
	const int batchWidth = targetBlob->GetBatchWidth();
	const int classesCount = targetBlob->GetObjectSize();

	CArray<int> buffer;
	buffer.SetSize( batchWidth );
	inputLengths->CopyTo( buffer.GetPtr() );
	int minLength = maxInputLength;
	for( int i = 0; i < buffer.Size(); ++i ) {
		if( buffer[i] < minLength ) {
			minLength = buffer[i];
		}
	}
	for( int timePosition = maxInputLength - 1; timePosition >= minLength; --timePosition ) {
		targetBlobWindow->SetParentPos( timePosition );
		for( int batchIndex = 0; batchIndex < batchWidth; ++batchIndex ) {
			if( buffer[batchIndex] <= timePosition ) {
				MathEngine().VectorCopy( targetBlobWindow->GetObjectData( batchIndex ), paddingBlob->GetData(), classesCount );
			}
		}
	}
}

void CCtcLossNaiveLayer::RunOnce()
{
	CDnnBlob* labelsLengths =
		inputBlobs.Size() <= I_LabelsLengths ? 0 : inputBlobs[I_LabelsLengths];
	CDnnBlob* inputLengths =
		inputBlobs.Size() <= I_InputLengths ? 0 : inputBlobs[I_InputLengths];

	const int batchWidth = inputBlobs[I_Labels]->GetBatchWidth();

	if( inputBlobs.Size() > I_LabelWeights ) {
		weights = inputBlobs[I_LabelWeights];
	}
	// Insert spaces
	CIntHandleStackVar fillValue( MathEngine() );
	fillValue.SetValue(blankLabel);
	MathEngine().MatrixSpreadRows(inputBlobs[I_Labels]->GetData<int>(), 
		inputBlobs[I_Labels]->GetBatchLength(), inputBlobs[I_Labels]->GetBatchWidth(),
		paddedLabels->GetData<int>(), paddedLabels->GetBatchLength(), 
		labelRows->GetData<int>(), fillValue);

	if( allowBlankLabelSkip ) {
		calculateBlankSkipMasks();
	}

	// Calculate the log(softmax) for the input results
	MathEngine().MatrixSoftmaxByRows(inputBlobs[I_Result]->GetData(), 
		inputBlobs[I_Result]->GetObjectCount(), inputBlobs[I_Result]->GetObjectSize(), 
		resultProb->GetData());
	MathEngine().VectorLog(resultProb->GetData(), resultLogProb->GetData(), 
		resultLogProb->GetDataSize());

	// Fill the ends with blanks
	paddingResultValue->Fill( logZero );
	paddingResultValue->GetData().SetValueAt( blankLabel, logOneNeg );
	applyInputLengthsPadding( inputLengths, paddingResultValue, resultLogProb, resultLogProbWindow );

	// Calculate the logarithm of prefix probability with a forward pass
	calculateForwardVariables();
	// Calculate the logarithm of suffix probability with a backward pass
	calculateBackwardVariables( labelsLengths, inputLengths );
	
	CFloatHandleStackVar totalLogProb(MathEngine(), batchWidth);
	// Maximize the total logarithm of the probability of recognizing the correct sequence
	logAlphaWindow->SetParentPos(0);
	logBetaWindow->SetParentPos(0);
	MathEngine().VectorAdd(logAlphaWindow->GetData(), logBetaWindow->GetData(), 
		logAlphaBeta->GetData(), logAlphaBeta->GetDataSize());

	// Calculate the total logarithm of the probability of recognizing
	// the correct sequence by adding across all possible pairings
	NeoAssert(logAlphaBeta->GetObjectSize() == batchWidth);
	matrixLogSumExpByColumns(*logAlphaBeta, logAlphaBeta->GetBatchWidth(), logAlphaBeta->GetObjectSize(), totalLogProb);

	// Take weights into account
	MathEngine().VectorDotProduct(weights->GetData(), totalLogProb, batchWidth,
		loss->GetData());
	// lossDivider is negative, so loss = -totalLogProb
	MathEngine().VectorMultiply(loss->GetData(), loss->GetData(), 1,
		lossDivider->GetData());

	// Calculate the loss function gradient
	if(IsBackwardPerformed()) {
		calculateGradient(totalLogProb);
	}
}

void CCtcLossNaiveLayer::BackwardOnce()
{
	// Take weights into account
	MathEngine().Multiply1DiagMatrixByMatrix( lossGradient->GetBatchLength(), weights->GetData(),
		lossGradient->GetBatchWidth(), lossGradient->GetData(), lossGradient->GetObjectSize(),
		inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize() );
	MathEngine().VectorMultiply(inputDiffBlobs[I_Result]->GetData(), 
		inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize(), 
		lossGradientDivider->GetData());
	// In case of "huge" gradients the system behavior may be incorrect,
	// so cut these values down
	MathEngine().VectorMinMax(inputDiffBlobs[I_Result]->GetData(), 
		inputDiffBlobs[I_Result]->GetData(), inputDiffBlobs[I_Result]->GetDataSize(), 
		minGradient->GetData(), maxGradient->GetData());
}

// ====================================================================================================================

class CDummyLearn : public CBaseLayer
{
	NEOML_DNN_LAYER( CDummyLearn )
public:
	explicit CDummyLearn( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CDummyLearn", true ) {}

	CPtr<CDnnBlob> ActualDiff;

	void Serialize( CArchive& /* archive */ ) override { NeoAssert( false ); }

protected:
	void Reshape() override { outputDescs[0] = inputDescs[0]; }

	void RunOnce() override { outputBlobs[0]->CopyFrom( inputBlobs[0] ); }

	void BackwardOnce() override { inputDiffBlobs[0]->CopyFrom( outputDiffBlobs[0] ); }

	void LearnOnce() override
	{
		if( ActualDiff != nullptr && ActualDiff->HasEqualDimensions( outputDiffBlobs[0] ) ) {
			ActualDiff->CopyFrom( outputDiffBlobs[0] );
		} else {
			ActualDiff = outputDiffBlobs[0]->GetCopy();
		}
	}

	int BlobsForBackward() const override { return 0; }
	int BlobsForLearn() const override { return 0; }
};

// ====================================================================================================================

class CCtcTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CTestParams> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

} // namespace NeoMLTest

// ====================================================================================================================

using namespace NeoML;
using namespace NeoMLTest;

static void normalizeData( int classCount, CArray<float>& data )
{
	NeoAssert( data.Size() % classCount == 0 );
	const int objectCount = data.Size() / classCount;
	for( int obj = 0; obj < objectCount; ++obj ) {
		float total = 0.f;
		const int baseIndex = obj * classCount;
		for( int c = 0; c < classCount; ++c ) {
			total += data[baseIndex + c];
		}
		for( int c = 0; c < classCount; ++c ) {
			data[baseIndex + c] /= total;
		}
	}
}

template<class T>
static CPtr<CSourceLayer> addSourceLayer( const CString& name, int batchLen, int batchWidth, int classCount,
	const CArray<T>& data, CDnn& dnn )
{
	NeoAssert( data.Size() == batchLen * batchWidth * classCount );
	CPtr<CSourceLayer> source = new CSourceLayer( MathEngine() );
	source->SetName( name );
	CPtr<CDnnBlob> blob = CDnnBlob::CreateDataBlob( MathEngine(), CBlobType<T>::GetType(), batchLen, batchWidth, classCount );
	blob->CopyFrom( data.GetPtr() );
	dnn.AddLayer( *source );
	source->SetBlob( blob );
	return source;
}

static const char* dummyLearnName = "DummyLearn";

template<class CTC>
static CPtr<CTC> buildDnn( int resultLen, int batchSize, int classCount, int labelLen, int blankLabel, bool skipBlanks, float lossWeight,
	const CArray<float>& resultData, const CArray<int>& labelData, const CArray<int>* labelLenData,
	const CArray<int>* resultLenData, const CArray<float>* weightData, CDnn& dnn )
{
	CPtr<CSourceLayer> resultSource = addSourceLayer( "Result", resultLen, batchSize, classCount, resultData, dnn );
	CPtr<CDummyLearn> dummyLearn = AddLayer<CDummyLearn>( dummyLearnName, { resultSource.Ptr() } );
	CPtr<CSourceLayer> labelSource = addSourceLayer( "Label", labelLen, batchSize, 1, labelData, dnn );
	CPtr<CTC> ctc = AddLayer<CTC>( "Ctc", { dummyLearn.Ptr(), labelSource.Ptr() } );
	ctc->SetBlankLabel( blankLabel );
	ctc->SetAllowBlankLabelSkips( skipBlanks );
	ctc->SetLossWeight( lossWeight );

	if( labelLenData != nullptr ) {
		CPtr<CSourceLayer> labelLenSource = addSourceLayer( "LabelLen", 1, batchSize, 1, *labelLenData, dnn );
		ctc->Connect( 2, *labelLenSource );
	}

	if( resultLenData != nullptr ) {
		CPtr<CSourceLayer> resultLenSource = addSourceLayer( "ResultLen", 1, batchSize, 1, *resultLenData, dnn );
		ctc->Connect( 3, *resultLenSource );
	}

	if( weightData != nullptr ) {
		CPtr<CSourceLayer> weightSource = addSourceLayer( "Weight", 1, batchSize, 1, *weightData, dnn );
		ctc->Connect( 4, *weightSource );
	}

	return ctc;
}

namespace FObj {
	inline void swap( FObj::CArray<int>*& a, FObj::CArray<int>*& b ) {
		std::swap( a, b );
	}
	inline void swap( FObj::CArray<float>*& a, FObj::CArray < float >*& b ) {
		std::swap( a, b );
	}
}

static void ctcTestImpl( const CTestParams& params, int seed )
{
	CRandom random( seed );

	const CInterval batchSizeInterval = params.GetInterval( "BatchSize" );
	const CInterval labelLenInterval = params.GetInterval( "LabelLen" );
	const CInterval classCountInterval = params.GetInterval( "ClassCount" );

	const int batchSize = random.UniformInt( batchSizeInterval.Begin, batchSizeInterval.End );
	const int labelLen = random.UniformInt( labelLenInterval.Begin, labelLenInterval.End );
	const int resultLen = random.UniformInt( labelLen * 2 + 1, labelLen * 4 );
	const int classCount = random.UniformInt( classCountInterval.Begin, classCountInterval.End );
	const int blankLabel = random.UniformInt( 0, classCount - 1 );
	const bool skipBlanks = random.Next() % 2 == 0;
	const float lossWeight = random.Next() % 2 == 0 ? 1.f : static_cast<float>( random.Uniform( 0.1, 0.9 ) );

	CREATE_FILL_FLOAT_ARRAY( result, 0.f, 1.f, resultLen * batchSize * classCount, random );
	normalizeData( classCount, result );
	CREATE_FILL_INT_ARRAY( label, 0, classCount - 1, labelLen * batchSize, random );

	std::unique_ptr<CArray<int>> labelLens;
	std::unique_ptr<CArray<int>> resultLens;
	std::unique_ptr<CArray<float>> weights;

	if( random.Next() % 2 == 0 ) {
		labelLens.reset( new CArray<int>() );
		labelLens->SetBufferSize( batchSize );
		for( int i = 0; i < batchSize; ++i ) {
			labelLens->Add( random.UniformInt( 1, labelLen ) );
		}
		if( random.Next() % 2 == 0 ) {
			resultLens.reset( new CArray<int>() );
			resultLens->SetBufferSize( batchSize );
			for( int i = 0; i < batchSize; ++i ) {
				resultLens->Add( random.UniformInt( ( *labelLens )[i] * 2 + 1, resultLen ) );
			}
			if( random.Next() % 2 == 0 ) {
				weights.reset( new CArray<float>() );
				weights->SetBufferSize( batchSize );
				for( int i = 0; i < batchSize; ++i ) {
					weights->Add( static_cast<float>( random.Uniform( 0., 2. ) ) );
				}
			}
		}
	}

	CDnn naiveDnn( random, MathEngine() );
	CPtr<CCtcLossNaiveLayer> naiveLoss = buildDnn<CCtcLossNaiveLayer>( resultLen, batchSize, classCount, labelLen,
		blankLabel, skipBlanks, lossWeight,
		result, label, labelLens.get(), resultLens.get(), weights.get(), naiveDnn );
	CPtr<CDummyLearn> naiveLearn = CheckCast<CDummyLearn>( naiveDnn.GetLayer( dummyLearnName ) );

	CDnn actualDnn( random, MathEngine() );
	CPtr<CCtcLossLayer> actualLoss = buildDnn<CCtcLossLayer>( resultLen, batchSize, classCount, labelLen,
		blankLabel, skipBlanks, lossWeight,
		result, label, labelLens.get(), resultLens.get(), weights.get(), actualDnn );
	CPtr<CDummyLearn> actualLearn = CheckCast<CDummyLearn>( actualDnn.GetLayer( dummyLearnName ) );

	naiveDnn.RunAndBackwardOnce();
	actualDnn.RunAndBackwardOnce();

	EXPECT_TRUE( FloatEq( naiveLoss->GetLastLoss(), actualLoss->GetLastLoss(), 1e-4f ) ) << naiveLoss->GetLastLoss()
		<< '\t' << actualLoss->GetLastLoss();
	EXPECT_TRUE( CompareBlobs( *naiveLoss->GetLastGradient(), *actualLoss->GetLastGradient(), 1e-4f ) );
	EXPECT_TRUE( CompareBlobs( *naiveLearn->ActualDiff, *actualLearn->ActualDiff, 1e-4f ) );
}

TEST_P( CCtcTest, Random )
{
	if( MathEngine().GetType() == MET_Cpu || MathEngine().GetType() == MET_Cuda ) {
		RUN_TEST_IMPL( ctcTestImpl );
	}
}

INSTANTIATE_TEST_CASE_P( CCtcTestInstantiation, CCtcTest,
	::testing::Values(
		CTestParams(
			"BatchSize=(1..1);"
			"LabelLen=(1..3);"
			"ClassCount=(2..80);"
			"TestCount=100;"
		),
		CTestParams(
			"BatchSize=(1..20);"
			"LabelLen=(1..30);"
			"ClassCount=(2..20);"
			"TestCount=100;"
		)
	)
);
