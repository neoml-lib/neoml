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

#include <NeoML/Dnn/Layers/CenterLossLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CCenterLossLayer::CCenterLossLayer( IMathEngine& mathEngine ) :
	CLossLayer( mathEngine, "CCnnCenterLossLayer" ),
	numberOfClasses( 0 ),
	classCentersConvergenceRate( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) ),
	oneMult( CDnnBlob::CreateVector( mathEngine, CT_Float, 1 ) )
{
	classCentersConvergenceRate->GetData().SetValue( 0.0f );
	oneMult->GetData().SetValue( 1.f );
}

static const int CenterLossLayerVersion = 2000;

void CCenterLossLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CenterLossLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CLossLayer::Serialize( archive );

	archive.Serialize( numberOfClasses );

	if( archive.IsStoring() ) {
		archive << GetClassCentersConvergenceRate();
	} else if( archive.IsLoading() ) {
		float tmp;
		archive >> tmp;
		SetClassCentersConvergenceRate( tmp );
	} else {
		NeoAssert( false );
	}
}

void CCenterLossLayer::Reshape()
{
	CLossLayer::Reshape();
	CheckLayerArchitecture( inputDescs[1].GetDataType() == CT_Int, "labels must be CT_Int" );
	CheckLayerArchitecture( inputDescs[1].ObjectSize() == 1, "should be one number for one label" );
}

void CCenterLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstIntHandle label, int /* labelSize */, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	// The total input size
	const int inputDataSize = batchSize * vectorSize;

	if(classCentersBlob == nullptr) {
		classCentersBlob = CDnnBlob::CreateMatrix(MathEngine(), CT_Float, numberOfClasses, vectorSize);
		classCentersBlob->Fill( 0.f );
	}
	// The current class centers
	CConstFloatHandle classCenters = classCentersBlob->GetData();
	// Remember the difference between the input features and the current class centers 
	// for these objects according to their labels: x_i - c_{y_i}
	CFloatHandleStackVar tempDiff( MathEngine(), inputDataSize );

	// Copy the current center values for the input classes
	CLookupDimension lookupDimension( numberOfClasses, vectorSize );
	MathEngine().VectorMultichannelLookupAndCopy( batchSize, 1, label, &classCenters, &lookupDimension, 1,
		tempDiff, vectorSize );
	// Remember the difference between the calculated features and the current centers for these objects
	MathEngine().VectorSub( data, tempDiff, tempDiff, inputDataSize );

	// When not learning, that is, running the network to get the current loss value,
	// there is no need to calculate loss gradient and update the centers
	if( !lossGradient.IsNull() ) {
		// The x_i - c_{y_i} value is the same as derivative by the inputs
		MathEngine().VectorCopy( lossGradient, tempDiff, inputDataSize );
		// Update the class centers
		updateCenters( tempDiff );
	}

	CFloatHandle tempDiffSquared = tempDiff;
	// Calculate the squared difference from above and the error on the elements
	MathEngine().VectorEltwiseMultiply( tempDiff, tempDiff, tempDiffSquared, inputDataSize );
	MathEngine().SumMatrixColumns( lossValue, tempDiffSquared, batchSize, vectorSize );
}

// Update the class centers on the backward pass using the current batch data
void CCenterLossLayer::updateCenters( const CConstFloatHandle& tempDiff )
{
	const int inputSize = inputBlobs[0]->GetDataSize();
	const int objectCount = inputBlobs[0]->GetObjectCount();
	const int numberOfFeatures = inputBlobs[0]->GetObjectSize();
	const int classCentersSize = classCentersBlob->GetDataSize();

	CFloatHandleStackVar temp( MathEngine(), classCentersSize * 2 + inputSize );

	CFloatHandle classCentersNumerator = temp;
	CFloatHandle classCentersDenominator = temp + classCentersSize;
	CFloatHandle onesTempBlob = temp + classCentersSize * 2;
	CFloatHandle handlesArray[1];

	CFloatHandle classCenters = classCentersBlob->GetData();
	CConstIntHandle labels = inputBlobs[1]->GetData<int>();

	// The numerator of the correction: the total of x_i - c_{y_i}, aggregated by classes
	MathEngine().VectorFill(classCentersNumerator, 0.0f, classCentersSize);
	handlesArray[0] = classCentersNumerator;

	CLookupDimension lookupDimension( /*count*/numberOfClasses, /*size*/numberOfFeatures );
	MathEngine().VectorMultichannelLookupAndAddToTable( objectCount, 1, labels,
		handlesArray, &lookupDimension, 1, oneMult->GetData(), tempDiff, numberOfFeatures );

	MathEngine().VectorFill( onesTempBlob, 1.0f, inputSize );
	// The denominator of the correction: 1 + the number of elements of this class in the batch
	MathEngine().VectorFill(classCentersDenominator, 1.0f, classCentersSize);
	handlesArray[0] = classCentersDenominator;

	MathEngine().VectorMultichannelLookupAndAddToTable( objectCount, 1, labels,
		handlesArray, &lookupDimension, 1, oneMult->GetData(), onesTempBlob, numberOfFeatures );

	// The final correction = \alpha * numerator / denominator
	MathEngine().VectorEltwiseDivide( classCentersNumerator, classCentersDenominator,
		classCentersNumerator, classCentersSize );
	MathEngine().VectorMultiply( classCentersNumerator, classCentersNumerator,
		classCentersSize, classCentersConvergenceRate->GetData() );
	MathEngine().VectorAdd( classCenters, classCentersNumerator, classCenters,
		classCentersSize );
}

CLayerWrapper<CCenterLossLayer> CenterLoss(
	int numberOfClasses, float classCentersConvergenceRate, float lossWeight )
{
	return CLayerWrapper<CCenterLossLayer>( "CenterLoss", [=]( CCenterLossLayer* result ) {
		result->SetNumberOfClasses( numberOfClasses );
		result->SetClassCentersConvergenceRate( classCentersConvergenceRate );
		result->SetLossWeight( lossWeight );
	} );
}

} // namespace NeoML
