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
	positivesTotal( 0 ),
	negativesTotal( 0 ),
	positivesCorrect( 0 ),
	negativesCorrect( 0 ),
	params( CDnnBlob::CreateVector( mathEngine, CT_Float, TP_Count_ ) )
{
	params->Clear();
}

void CPrecisionRecallLayer::Reshape()
{
	CQualityControlLayer::Reshape();
	// Intended for binary classification
	// For multi-class classificiation use AccuracyLayer
	NeoAssert( inputDescs[0].Channels() == 1 && inputDescs[0].Height() == 1
		&& inputDescs[0].Width() == 1 );
	NeoAssert( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount() );
	NeoAssert( inputDescs[0].ObjectSize() >= 1 );
	NeoAssert( inputDescs[1].Channels() == 1 && inputDescs[1].Height() == 1
		&& inputDescs[1].Width() == 1 );

	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_Channels, 4 );
}

void CPrecisionRecallLayer::GetLastResult( CArray<int>& results )
{
	results.FreeBuffer();
	results.Add( PositivesCorrect() );
	results.Add( PositivesTotal() );
	results.Add( NegativesCorrect() );
	results.Add( NegativesTotal() );
}

void CPrecisionRecallLayer::OnReset()
{
	PositivesCorrect() = 0;
	PositivesTotal() = 0;
	NegativesCorrect() = 0;
	NegativesTotal() = 0;
}

void CPrecisionRecallLayer::RunOnceAfterReset()
{
	CConstFloatHandle calculatedLogit = inputBlobs[0]->GetData();
	CConstFloatHandle groundtruth = inputBlobs[1]->GetData();

	const int vectorSize = inputBlobs[0]->GetDataSize();
	CFloatHandleStackVar tempVar( MathEngine(), vectorSize * 3 );
	CFloatHandle binarizedLabel = tempVar.GetHandle() + vectorSize;
	CFloatHandle binarizedCalculation = tempVar.GetHandle() + 2 * vectorSize;

	CFloatHandleStackVar zero( MathEngine() );
	zero.SetValue( 0.f );
	CFloatHandleStackVar minusOne( MathEngine() );
	minusOne.SetValue( -1.f );

	{
		CFloatHandle ones = tempVar.GetHandle(); // reduced memory usage for calculation
		MathEngine().VectorFill( ones, 1.0f, vectorSize );
		// Mask of the elements classified as +1 class (logits are positive)
		MathEngine().VectorReLUDiff( calculatedLogit, ones, binarizedCalculation, vectorSize, zero );
		// Mask of the elements whose ground truth is +1
		MathEngine().VectorReLUDiff( groundtruth, ones, binarizedLabel, vectorSize, zero );
	}
	{
		// Mask of the correctly classified +1 objects
		CFloatHandle truePositives = tempVar.GetHandle(); // reduced memory usage for calculation
		// 1 only if corresponding numbers in both vectors are 1 (otherwise it's 0)
		MathEngine().VectorEltwiseMin( binarizedLabel, binarizedCalculation, truePositives, vectorSize );
		// Number of the correctly classified +1 objects
		MathEngine().VectorSum( truePositives, vectorSize, params->GetData( { TP_TruePositivesCount } ) );
	}
	{
		// Number of the +1 objects (ground truth)
		CFloatHandle temp = tempVar.GetHandle(); // reduced memory usage for calculation
		MathEngine().VectorCopy( temp, binarizedLabel, vectorSize );
		MathEngine().VectorSum( temp, vectorSize, params->GetData( { TP_PositivesCount } ) );
	}
	{
		// Mask of the correctly classified -1 objects
		CFloatHandle trueNegative = tempVar.GetHandle(); // reduced memory usage for calculation
		// 0 only if corresponding numbers in both vectors are 0 (otherwise 1)
		MathEngine().VectorEltwiseMax( binarizedLabel, binarizedCalculation, trueNegative, vectorSize );
		// At this moment true negative elements are marked as 0
		// Inverting this vector
		// {0, 1} -> {-1, 0}
		MathEngine().VectorAddValue( trueNegative, trueNegative, vectorSize, minusOne );
		// {-1, 0} -> {1, 0}
		MathEngine().VectorAbs( trueNegative, trueNegative, vectorSize );
		// Number of the correctly classified -1 objects
		MathEngine().VectorSum( trueNegative, vectorSize, params->GetData( { TP_TrueNegativeCount } ) );
	}
	// Number of the -1 objects (ground truth)
	// At this moment -1 objects are marked as 0
	// Inverting this vector
	// {0, 1} -> {-1, 0}
	MathEngine().VectorAddValue( binarizedLabel, binarizedLabel, vectorSize, minusOne );
	// {-1, 0} -> {1, 0}
	MathEngine().VectorAbs( binarizedLabel, binarizedLabel, vectorSize );
	MathEngine().VectorSum( binarizedLabel, vectorSize, params->GetData( { TP_NegativesCount } ) );
	{
		float buffer[TP_Count_];
		params->CopyTo( buffer );

		PositivesTotal() += to<int>( buffer[TP_PositivesCount] );
		NegativesTotal() += to<int>( buffer[TP_NegativesCount] );
		PositivesCorrect() += to<int>( buffer[TP_TruePositivesCount] );
		NegativesCorrect() += to<int>( buffer[TP_TrueNegativeCount] );
	}
	NeoAssert( PositivesTotal() >= 0 );
	NeoAssert( NegativesTotal() >= 0 );
	NeoAssert( PositivesCorrect() <= PositivesTotal() );
	NeoAssert( NegativesCorrect() <= NegativesTotal() );
	{
		float buffer[4]{
			static_cast<float>( PositivesCorrect() ),
			static_cast<float>( PositivesTotal() ),
			static_cast<float>( NegativesCorrect() ),
			static_cast<float>( NegativesTotal() )
		};
		outputBlobs[0]->CopyFrom( buffer );
	}
}

static const int PrecisionRecallLayerVersion = 2000;

void CPrecisionRecallLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( PrecisionRecallLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CQualityControlLayer::Serialize( archive );
}

CLayerWrapper<CPrecisionRecallLayer> PrecisionRecall()
{
	return CLayerWrapper<CPrecisionRecallLayer>( "PrecisionRecall" );
}

} // namespace NeoML
