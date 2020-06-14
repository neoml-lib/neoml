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

void CCenterLossLayer::BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize,
	CConstIntHandle label, int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient )
{
	// One number for one label
	NeoAssert( labelSize == 1 );

	// The total input size
	const int inputDataSize = batchSize * vectorSize;

	if(classCentersBlob == 0) {
		classCentersBlob = CDnnBlob::CreateMatrix(MathEngine(), CT_Float, numberOfClasses, vectorSize);
		classCentersBlob->Fill<float>( 0.f );
	}
	// The current class centers
	CConstFloatHandle classCenters = classCentersBlob->GetData<float>();
	// Remember the difference between the input features and the current class centers 
	// for these objects according to their labels: x_i - c_{y_i}
	CFloatHandleVar tempDiffHandle(MathEngine(), inputDataSize);

	// Copy the current center values for the input classes
	CLookupDimension lookupDimension;
	lookupDimension.VectorCount = numberOfClasses;
	lookupDimension.VectorSize = vectorSize;
	MathEngine().VectorMultichannelLookupAndCopy( batchSize, 1, label, &classCenters, &lookupDimension, 1,
		tempDiffHandle.GetHandle(), vectorSize );

	// Remember the difference between the calculated features and the current centers for these objects
	MathEngine().VectorSub( data, tempDiffHandle.GetHandle(), tempDiffHandle.GetHandle(), inputDataSize );

	// Calculate the squared difference from above and the error on the elements
	CFloatHandleVar diffSquared(MathEngine(), inputDataSize);
	MathEngine().VectorEltwiseMultiply( tempDiffHandle.GetHandle(), tempDiffHandle.GetHandle(), diffSquared.GetHandle(), inputDataSize );
	MathEngine().SumMatrixColumns( lossValue, diffSquared.GetHandle(), batchSize, vectorSize );

	// When not learning, that is, running the network to get the current loss value,
	// there is no need to calculate loss gradient and update the centers
	if( lossGradient.IsNull() ) {
		return;
	}
	// The x_i - c_{y_i} value is the same as derivative by the inputs
	MathEngine().VectorCopy( lossGradient, tempDiffHandle.GetHandle(), tempDiffHandle.Size() );

	// Update the class centers
	updateCenters( tempDiffHandle.GetHandle());
}

// Update the class centers on the backward pass using the current batch data
void CCenterLossLayer::updateCenters(const CFloatHandle& tempDiffHandle)
{
	const int objectCount = inputBlobs[0]->GetObjectCount();
	const int numberOfFeatures = inputBlobs[0]->GetObjectSize();

	CFloatHandle classCenters = classCentersBlob->GetData<float>();
	CConstIntHandle labels = inputBlobs[1]->GetData<int>();

	CLookupDimension lookupDimension;
	lookupDimension.VectorCount = numberOfClasses;
	lookupDimension.VectorSize = numberOfFeatures;
	CFloatHandle handlesArray[1];
	// The numerator of the correction: the total of x_i - c_{y_i}, aggregated by classes
	CFloatHandleVar classCentersUpdatesNumerator(MathEngine(), classCentersBlob->GetDataSize());
	MathEngine().VectorFill(classCentersUpdatesNumerator.GetHandle(), 0.0f, classCentersUpdatesNumerator.Size());
	handlesArray[0] = classCentersUpdatesNumerator.GetHandle();

	MathEngine().VectorMultichannelLookupAndAddToTable( objectCount, 1, labels, 
		handlesArray, &lookupDimension, 1, oneMult->GetData(), tempDiffHandle, numberOfFeatures );

	CFloatHandleVar onesTemporaryBlob(MathEngine(), inputBlobs[0]->GetDataSize());
	MathEngine().VectorFill(onesTemporaryBlob.GetHandle(), 1.0f, onesTemporaryBlob.Size());
	// The denominator of the correction: 1 + the number of elements of this class in the batch
	CFloatHandleVar classCentersUpdatesDenominator(MathEngine(), classCentersBlob->GetDataSize());
	MathEngine().VectorFill(classCentersUpdatesDenominator.GetHandle(), 1.0f, classCentersUpdatesDenominator.Size());
	handlesArray[0] = classCentersUpdatesDenominator.GetHandle();

	MathEngine().VectorMultichannelLookupAndAddToTable( objectCount, 1, labels, 
		handlesArray, &lookupDimension, 1, oneMult->GetData(), onesTemporaryBlob.GetHandle(), numberOfFeatures );

	// The final correction = \alpha * numerator / denominator
	MathEngine().VectorEltwiseDivide( classCentersUpdatesNumerator.GetHandle(), classCentersUpdatesDenominator.GetHandle(),
		classCentersUpdatesNumerator.GetHandle(), classCentersBlob->GetDataSize() );
	MathEngine().VectorMultiply( classCentersUpdatesNumerator.GetHandle(), classCentersUpdatesNumerator.GetHandle(),
		classCentersBlob->GetDataSize(), classCentersConvergenceRate->GetData() );
	MathEngine().VectorAdd( classCenters, classCentersUpdatesNumerator.GetHandle(), classCenters,
		classCentersBlob->GetDataSize() );
}

} // namespace NeoML
