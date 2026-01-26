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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/PrecisionRecallLayer.h>

namespace NeoML {

CPrecisionRecallLayer::CPrecisionRecallLayer( IMathEngine& mathEngine ) :
	CQualityControlLayer( mathEngine, "CCnnPrecisionRecallLayer" ),
	accumulated( CDnnBlob::CreateVector( mathEngine, CT_Int, TP_Count ) ),
	current( CDnnBlob::CreateVector( mathEngine, CT_Int, TP_Count ) )
{
	accumulated->Clear();
	current->Clear();
}

void CPrecisionRecallLayer::Reshape()
{
	CQualityControlLayer::Reshape();
	// Intended for binary classification
	// For multi-class classificiation use AccuracyLayer
	NeoAssert( inputDescs[0].Channels() == 1 && inputDescs[0].Height() == 1 && inputDescs[0].Width() == 1 );
	NeoAssert( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount() );
	NeoAssert( inputDescs[0].ObjectSize() >= 1 );
	NeoAssert( inputDescs[1].Channels() == 1 && inputDescs[1].Height() == 1 && inputDescs[1].Width() == 1 );

	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_Channels, 4 );
}

void CPrecisionRecallLayer::GetLastResult( CArray<int>& results ) const
{
	results.SetSize( TP_Count );
	accumulated->CopyTo( results.GetPtr() ); // sync
}

void CPrecisionRecallLayer::OnReset()
{
	accumulated->Clear();
	current->Clear();
}

void CPrecisionRecallLayer::RunOnceAfterReset()
{
	CConstFloatHandle calculatedLogit = inputBlobs[0]->GetData();
	CConstFloatHandle groundTruth = inputBlobs[1]->GetData();

	const int vectorSize = inputBlobs[0]->GetDataSize();
	CFloatHandleStackVar temp( MathEngine(), vectorSize * 3 + TP_Count + 2 );

	CFloatHandle binarizedLabel = temp.GetHandle() + vectorSize;
	CFloatHandle binarizedCalculation = binarizedLabel + vectorSize;
	CFloatHandle params = binarizedCalculation + vectorSize;

	CFloatHandle zero = params + TP_Count;
	zero.SetValue( 0.f );
	CFloatHandle minusOne = zero + 1;
	minusOne.SetValue( -1.f );

	{
		CFloatHandle ones = temp.GetHandle(); // reduced memory usage for calculation

		MathEngine().VectorFill( ones, 1.0f, vectorSize );
		// Mask of the elements classified as +1 class (logits are positive)
		MathEngine().VectorReLUDiff( calculatedLogit, ones, binarizedCalculation, vectorSize, zero );
		// Mask of the elements whose ground truth is +1
		MathEngine().VectorReLUDiff( groundTruth, ones, binarizedLabel, vectorSize, zero );

		// Number of the +1 objects (ground truth)
		MathEngine().VectorSum( binarizedLabel, vectorSize, params + TP_PositivesTotal );
	}
	{
		CFloatHandle truePositives = temp.GetHandle(); // reduced memory usage for calculation

		// Mask of the correctly classified +1 objects
		// 1 only if corresponding numbers in both vectors are 1 (otherwise it's 0)
		MathEngine().VectorEltwiseMin( binarizedLabel, binarizedCalculation, truePositives, vectorSize );
		// Number of the correctly classified +1 objects
		MathEngine().VectorSum( truePositives, vectorSize, params + TP_PositivesCorrect );
	}
	{
		CFloatHandle trueNegative = temp.GetHandle(); // reduced memory usage for calculation
		CFloatHandle negativesCorrect = params + TP_NegativesCorrect;

		// Mask of the correctly classified -1 objects
		// 0 only if corresponding numbers in both vectors are 0 (otherwise 1)
		MathEngine().VectorEltwiseMax( binarizedLabel, binarizedCalculation, trueNegative, vectorSize );
		// At this moment true negative elements are marked as 0
		// Inverting this vector
		// {0, 1} -> {-1, 0}
		MathEngine().VectorAddValue( trueNegative, trueNegative, vectorSize, minusOne );
		// Number of the correctly classified -1 objects
		MathEngine().VectorSum( trueNegative, vectorSize, negativesCorrect );
		// {-1, 0} -> {1, 0}
		MathEngine().VectorAbs( negativesCorrect, negativesCorrect, 1 );
	}
	{
		CFloatHandle negativesTotal = params + TP_NegativesTotal;
		// Number of the -1 objects (ground truth)
		// At this moment -1 objects are marked as 0
		// Inverting this vector
		// {0, 1} -> {-1, 0}
		MathEngine().VectorAddValue( binarizedLabel, binarizedLabel, vectorSize, minusOne );
		MathEngine().VectorSum( binarizedLabel, vectorSize, negativesTotal );
		// {-1, 0} -> {1, 0}
		MathEngine().VectorAbs( negativesTotal, negativesTotal, 1 );
	}

	MathEngine().VectorConvert( params, current->GetData<int>(), TP_Count );
	MathEngine().VectorAdd( current->GetData<int>(), accumulated->GetData<int>(), accumulated->GetData<int>(), TP_Count );

	NeoPresume( PositivesTotal() >= 0 ); // sync
	NeoPresume( NegativesTotal() >= 0 ); // sync
	NeoPresume( PositivesCorrect() <= PositivesTotal() ); // sync
	NeoPresume( NegativesCorrect() <= NegativesTotal() ); // sync

	MathEngine().VectorConvert( accumulated->GetData<int>(), outputBlobs[0]->GetData(), TP_Count );
}

constexpr int precisionRecallLayerVersion = 2000;

void CPrecisionRecallLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( precisionRecallLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CQualityControlLayer::Serialize( archive );
}

CLayerWrapper<CPrecisionRecallLayer> PrecisionRecall()
{
	return CLayerWrapper<CPrecisionRecallLayer>( "PrecisionRecall" );
}

} // namespace NeoML
