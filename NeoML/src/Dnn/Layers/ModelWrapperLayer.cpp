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

#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/ModelWrapperLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Layers/SourceLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>
#include <NeoML/TraditionalML/Shuffler.h>

namespace NeoML {

// Shuffles the elements array.
static void shuffle( CArray<int>& elements, CRandom& random )
{
	CShuffler indexGenerator( random, elements.Size() );
	CArray<int> oldElements;
	elements.CopyTo( oldElements );
	for( int i = 0; i < elements.Size(); ++i ) {
		elements[i] = oldElements[indexGenerator.Next()];
	}
}

//---------------------------------------------------------------------------------------------------------------------

// Shuffles the elements of an array and returns them one by one.
// Unlike CShuffler, it returns the elements of the array, not the indices of the elements, 
// and also supports cyclicity (when the end of the sequence is reached, it will be shuffled again).
template <typename T>
class CShuffledElements final {
public:
	CShuffledElements( CArray<T>&& _elements, CRandom& _random ) :
		random( _random ), elements( std::move( _elements ) ), shuffler( random, elements.Size() ) {}
	CShuffledElements( CShuffledElements&& ) = default;

	CShuffledElements& operator=( CShuffledElements&& ) = default;

	// The number of elements in the elements array.
	int Size() const { return elements.Size(); }
	// Generates the next element of the sequence.
	T Next() { if( shuffler.IsFinished() ) { shuffler.Reset(); } return elements[shuffler.Next()]; }

private:
	CRandom& random; // The random generator
	CArray<T> elements; // The shuffled elements array
	CShuffler shuffler; // The index shuffled generator
};

//---------------------------------------------------------------------------------------------------------------------

// Forms a batch in such a way that it contains the same number of positive and negative pairs, and
// all positive elements are collected from one class, and all negative examples are from different non-intersecting ones.
// For example, let there be 4 classes and batchSize = 12.
// The number of matchings in the batch is C_12^2 = 66, respectively, the number of positive and negative pairs is 33 each.
// We sample a random "positive" class from which we will collect positive elements, let's say this is class 0.
// Then we randomly sample 9 elements from class 0, because C_9^2 = 36 pairs are closest to 33 (for
// comparison, 8 elements give 28 pairs). The remaining 12 - 9 = 3 elements are taken each from a separate class - 
// from the 1st, 2nd and 3rd classes, respectively.
// As a result, we obtained a batch, the matching elements of which give 36 positive and 30 negative pairs.
class CBalancedPairBatchGenerator : public IShuffledBatchGenerator {
public:
	CBalancedPairBatchGenerator( const IProblem& problem, unsigned seed );

	// IShuffledBatchGenerator
	const CArray<int>& GenerateBatchIndexes( int batchSize, bool batchShuffled ) override;
	bool HasUnseenElements() const override { return !unseenElementsIndices.IsEmpty(); }
	void DeleteUnseenElement( int index ) override { unseenElementsIndices.Delete( index ); }

private:
	CRandom random;
	// The map of "label -> this label's elements indexes generator".
	CMap<int, CShuffledElements<int>> labelToIndexGenerator;
	// The labels generator
	CShuffledElements<int> labelGenerator;
	// Indexes of elements, which aren't in any batch, to detect epoch's end
	CHashTable<int> unseenElementsIndices;
	// Return value of indices set
	CArray<int> batchIndexes;

	CArray<int> getLabels( const IProblem& );
};

CBalancedPairBatchGenerator::CBalancedPairBatchGenerator( const IProblem& problem, unsigned seed ) :
	random( seed ),
	labelGenerator( getLabels( problem ), random )
{
	unseenElementsIndices.SetBufferSize( problem.GetVectorCount() );
	for( int i = 0; i < problem.GetVectorCount(); ++i ) {
		unseenElementsIndices.Add( i );
	}
}

CArray<int> CBalancedPairBatchGenerator::getLabels( const IProblem& problem )
{
	CMap<int, CArray<int>> labelToIndexes;
	for( int i = 0; i < problem.GetVectorCount(); ++i ) {
		labelToIndexes.GetOrCreateValue( problem.GetClass( i ) ).Add( i );
	}

	CArray<int> labelsUnique;
	labelsUnique.SetBufferSize( labelToIndexes.Size() );
	for( auto& item : labelToIndexes ) {
		labelToIndexGenerator.Add( item.Key, CShuffledElements<int>( std::move( item.Value ), random ) );
		labelsUnique.Add( item.Key );
	}
	return labelsUnique;
}

const CArray<int>& CBalancedPairBatchGenerator::GenerateBatchIndexes( int batchSize, bool batchShuffled )
{
	// The number of all matchings in the batch.
	const int numOfCombinations = batchSize * ( batchSize - 1 ) / 2;
	// The desired number of positive matchings in the batch.
	const int targetNumOfPositiveCombinations = numOfCombinations / 2;
	// How many elements of one class should be taken to get the desired number of positive matchings.
	// The formula below is one of the roots of the square equation C_n^2 = targetNumOfPositiveCombinations with respect to n.
	const double idealNumOfSingleClassSamples = 1.0 / 2.0 + sqrt( 1.0 / 4.0 + 2.0 * targetNumOfPositiveCombinations );
	// Round to the nearest integer - this will be the best approximation of the number of elements from one class.
	const int numOfSingleClassSamples = static_cast<int>( round( idealNumOfSingleClassSamples ) );

	// Sample a random class and collect numOfSingleClassSamples elements from it into a batch.
	const int majorLabel = labelGenerator.Next();
	NeoAssert( numOfSingleClassSamples <= labelToIndexGenerator.Get( majorLabel ).Size() );
	// Sample a random class and collect numOfSingleClassSamples elements from it into a batch.
	batchIndexes.DeleteAll();
	batchIndexes.SetBufferSize( numOfSingleClassSamples );
	for( int i = 0; i < numOfSingleClassSamples; ++i ) {
		batchIndexes.Add( labelToIndexGenerator.Get( majorLabel ).Next() );
	}

	// The number of remaining elements, also the number of remaining classes, because for the remaining elements
	// the equality "one element = one class" is satisfied.
	const int numOfOtherClasses = batchSize - numOfSingleClassSamples;
	// Important! It is expected that all remaining elements in the batch will be from DIFFERENT clusters,
	// so their number should not exceed the total number of clusters, taking into account one already sampled.
	// If this is not the case, then we fail - let the user add more classes or reduce the batch size.
	NeoAssert( numOfOtherClasses + 1 <= labelToIndexGenerator.Size() );
	// Table of used classes to skip duplicates.
	CHashTable<int> usedClasses;
	usedClasses.Add( majorLabel );
	// Will never loop if numOfOtherClasses + 1 <= totalNumOfClasses, because the generator shuffles classes without duplicates.
	while( usedClasses.Size() < numOfOtherClasses + 1 ) {
		// Sample the class.
		const int label = labelGenerator.Next();
		// Skip the class if it has already been sampled.
		if( usedClasses.Has( label ) ) {
			continue;
		}
		usedClasses.Add( label );
		// Sample an element from this class.
		const int negativeSample = labelToIndexGenerator.Get( label ).Next();
		batchIndexes.Add( negativeSample );
	}
	if( batchShuffled ) {
		// Just in case, the batch is mixed before giving it to the model
		// So that it doesn’t accidentally learn the structure of the batch.
		shuffle( batchIndexes, random );
	}

	NeoAssert( batchIndexes.Size() == batchSize );
	return batchIndexes;
}

//---------------------------------------------------------------------------------------------------------------------

void CProblemSourceLayer::SetBatchSize(int _batchSize)
{
	if(_batchSize == batchSize) {
		return;
	}
	batchSize = _batchSize;
	ForceReshape();
}

void CProblemSourceLayer::SetProblem( const CPtr<const IProblem>& _problem, bool shuffle, unsigned seed )
{
	NeoAssert( _problem != nullptr );
	NeoAssert( GetDnn() == nullptr || problem == nullptr
		|| ( problem->GetFeatureCount() == _problem->GetFeatureCount()
			&& problem->GetClassCount() == _problem->GetClassCount() ) );

	problem = _problem;
	nextProblemIndex = 0;

	if( shuffle ) {
		shuffled = new CBalancedPairBatchGenerator( *problem, seed );
	}
}

void CProblemSourceLayer::SetLabelType( TBlobType newLabelType )
{
	NeoAssert( newLabelType == CT_Float || newLabelType == CT_Int );

	if( labelType == newLabelType ) {
		return;
	}

	labelType = newLabelType;

	ForceReshape();
}

void CProblemSourceLayer::Reshape()
{
	NeoAssert( !GetDnn()->IsRecurrentMode() );

	CheckLayerArchitecture( problem.Ptr() != nullptr, "source problem is null" );
	CheckOutputs();
	CheckLayerArchitecture( GetOutputCount() >= 2, "problem source layer has less than 2 outputs" );

	// The data
	outputDescs[EB_Data] = CBlobDesc( CT_Float );
	outputDescs[EB_Data].SetDimSize( BD_BatchWidth, batchSize );
	outputDescs[EB_Data].SetDimSize( BD_Channels, problem->GetFeatureCount() );
	exchangeBufs[EB_Data].SetSize( outputDescs[EB_Data].BlobSize() );

	// The labels
	int labelSize = problem->GetClassCount();
	if( labelSize == 2 ) {
		labelSize = 1;
	}
	outputDescs[EB_Label] = CBlobDesc( labelType );
	outputDescs[EB_Label].SetDimSize( BD_BatchWidth, batchSize );
	if( labelType != CT_Int ) {
		outputDescs[EB_Label].SetDimSize( BD_Channels, labelSize );
	}
	exchangeBufs[EB_Label].SetSize( outputDescs[EB_Label].BlobSize() );

	// The weights
	outputDescs[EB_Weight] = CBlobDesc( CT_Float );
	outputDescs[EB_Weight].SetDimSize( BD_BatchWidth, batchSize );
	exchangeBufs[EB_Weight].SetSize( outputDescs[EB_Weight].BlobSize() );
}

void CProblemSourceLayer::RunOnce()
{
	NeoAssert( problem != nullptr );

	::memset( exchangeBufs[EB_Label].GetPtr(), 0, exchangeBufs[EB_Label].Size() * sizeof( float ) );
	if( emptyFill == 0.f ) {
		::memset( exchangeBufs[EB_Data].GetPtr(), 0, exchangeBufs[EB_Data].Size() * sizeof( float ) );
	} else {
		for( int i = 0; i < exchangeBufs[EB_Data].Size(); ++i ) {
			exchangeBufs[EB_Data][i] = emptyFill;
		}
	}

	if( shuffled != nullptr ) {
		const CArray<int>& batchIndexes = shuffled->GenerateBatchIndexes( batchSize, /*batchShuffled*/true );
		for( int i = 0; i < batchSize; ++i ) {
			shuffled->DeleteUnseenElement( batchIndexes[i] );
			fillExchangeBuffers( i, batchIndexes[i] );
		}
	} else {
		const int vectorCount = problem->GetVectorCount();
		for( int i = 0; i < batchSize; ++i, ++nextProblemIndex ) {
			fillExchangeBuffers( i, nextProblemIndex % vectorCount );
		}
	}

	outputBlobs[EB_Data]->CopyFrom( exchangeBufs[EB_Data].GetPtr() );
	if( labelType == CT_Float ) {
		outputBlobs[EB_Label]->CopyFrom( exchangeBufs[EB_Label].GetPtr() );
	} else {
		outputBlobs[EB_Label]->CopyFrom( reinterpret_cast<int*>( exchangeBufs[EB_Label].GetPtr() ) );
	}
	outputBlobs[EB_Weight]->CopyFrom( exchangeBufs[EB_Weight].GetPtr() );
}

void CProblemSourceLayer::BackwardOnce()
{
	NeoAssert( false );
}

constexpr int ProblemSourceLayerVersion = 2001;

void CProblemSourceLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( ProblemSourceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	if( version >= 2001 ) {
		archive.Serialize( emptyFill );
	} else { // loading
		emptyFill = 0;
	}
	archive.Serialize( batchSize );
	int labelTypeInt = static_cast<int>( labelType );
	archive.Serialize( labelTypeInt );

	if( archive.IsLoading() ) {
		nextProblemIndex = NotFound;
		labelType = static_cast<TBlobType>( labelTypeInt );
		problem = nullptr;
		shuffled = nullptr;
	}
}

void CProblemSourceLayer::fillExchangeBuffers( int shift, int index )
{
	float* data = exchangeBufs[EB_Data].GetPtr() + shift * outputBlobs[EB_Data]->GetObjectSize();
	float* labels = exchangeBufs[EB_Label].GetPtr() + shift * outputBlobs[EB_Label]->GetObjectSize();
	float* weights = exchangeBufs[EB_Weight].GetPtr() + shift * outputBlobs[EB_Weight]->GetObjectSize();

	// The data
	const CFloatMatrixDesc matrix = problem->GetMatrix();
	CFloatVectorDesc vector;
	matrix.GetRow( index, vector );
	for( int j = 0; j < vector.Size; ++j ) {
		data[( vector.Indexes == nullptr ) ? j : vector.Indexes[j]] = static_cast<float>( vector.Values[j] );
	}

	// The labels
	// Update labels
	if( labelType == CT_Float ) {
		if( outputBlobs[EB_Label]->GetChannelsCount() == 1 ) {
			*labels = static_cast<float>( problem->GetBinaryClass( index ) );
		} else {
			const int classLabel = problem->GetClass( index );
			NeoAssert( 0 <= classLabel && classLabel < outputBlobs[EB_Label]->GetChannelsCount() );
			::memset( labels, 0, outputBlobs[EB_Label]->GetChannelsCount() * sizeof( float ) );
			labels[classLabel] = 1;
		}
	} else {
		static_assert( sizeof( float ) == sizeof( int ), "sizeof( float ) != sizeof( int )" );
		NeoAssert( outputBlobs[EB_Label]->GetChannelsCount() == 1 );
		*reinterpret_cast<int*>( labels ) = problem->GetClass( index );
	}

	// The weights
	*weights = static_cast<float>( problem->GetVectorWeight( index ) );
}

// Creates CProblemSourceLayer with the name
CProblemSourceLayer* ProblemSource( CDnn& dnn, const char* name,
	TBlobType labelType, int batchSize, const CPtr<const IProblem>& problem, bool shuffle, unsigned seed )
{
	CPtr<CProblemSourceLayer> result = new CProblemSourceLayer( dnn.GetMathEngine() );
	result->SetProblem( problem, shuffle, seed );
	result->SetLabelType( labelType );
	result->SetBatchSize( batchSize );
	result->SetName( name );
	dnn.AddLayer( *result );
	return result;
}

//---------------------------------------------------------------------------------------------------------------------

const char* const CDnnModelWrapper::SourceLayerName = "CCnnModelWrapper::SourceLayer";
const char* const CDnnModelWrapper::SinkLayerName = "CCnnModelWrapper::SinkLayer";

CDnnModelWrapper::CDnnModelWrapper(IMathEngine& _mathEngine, unsigned int seed) :
	ClassCount(0),
	SourceEmptyFill(0),	
	Random(seed),
	Dnn(Random, _mathEngine),
	mathEngine( _mathEngine )
{
	SourceLayer = FINE_DEBUG_NEW CSourceLayer( mathEngine );
	SourceLayer->SetName(SourceLayerName);

	SinkLayer = FINE_DEBUG_NEW CSinkLayer( mathEngine );
	SinkLayer->SetName(SinkLayerName);
}

bool CDnnModelWrapper::Classify(const CFloatVectorDesc& desc, CClassificationResult& result) const
{
	NeoAssert( SourceBlob != nullptr );
	NeoPresume( SourceBlob == SourceLayer->GetBlob() );

	exchangeBuffer.SetSize( SourceBlob->GetDataSize() );
	if( SourceEmptyFill == 0.f ) {
		::memset( exchangeBuffer.GetPtr(), 0, exchangeBuffer.Size() * sizeof( float ) );
	} else {
		for( int i = 0; i < exchangeBuffer.Size(); ++i ) {
			exchangeBuffer[i] = SourceEmptyFill;
		}
	}

	for(int i = 0; i < desc.Size; ++i) {
		exchangeBuffer[( desc.Indexes == nullptr ) ? i : desc.Indexes[i]] = desc.Values[i];
	}
	SourceBlob->CopyFrom( exchangeBuffer.GetPtr() );

	return classify( result );
}

constexpr int DnnModelWrapperVersion = 2001;

void CDnnModelWrapper::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( DnnModelWrapperVersion, CDnn::ArchiveMinSupportedVersion );

	archive.Serialize( ClassCount );
	if( version >= 2001 ) {
		archive.Serialize( SourceEmptyFill );
	} else { // loading
		SourceEmptyFill = 0;
	}
	archive.Serialize( Random );
	archive.Serialize( Dnn );

	CString sourceName = SourceLayer->GetName();
	archive.Serialize( sourceName );
	CString sinkName = SinkLayer->GetName();
	archive.Serialize( sinkName );

	CBlobDesc sourceDesc( CT_Float );
	sourceDesc.SetDimSize( BD_BatchWidth, 0 ); // set zero
	if( SourceBlob != nullptr ) {
		sourceDesc = SourceBlob->GetDesc();
	}
	for( int i = 0; i < CBlobDesc::MaxDimensions; ++i ) {
		int size = sourceDesc.DimSize( i );
		archive.Serialize( size );
		sourceDesc.SetDimSize( i, size );
	}

	if( archive.IsLoading() ) {
		if( Dnn.HasLayer( sourceName ) ) {
			SourceLayer = CheckCast<CSourceLayer>( Dnn.GetLayer( sourceName ).Ptr() );
		} else {
			SourceLayer->SetName( sourceName );
		}
		if( Dnn.HasLayer( sinkName ) ) {
			SinkLayer = CheckCast<CSinkLayer>( Dnn.GetLayer( sinkName ).Ptr() );
		} else {
			SinkLayer->SetName( sinkName );
		}
		if( sourceDesc.BlobSize() == 0 ) { // is zero
			SourceBlob = nullptr;
		} else {
			SourceBlob = CDnnBlob::CreateBlob(mathEngine, CT_Float, sourceDesc);
			SourceLayer->SetBlob(SourceBlob);
		}
		exchangeBuffer.SetSize(0);
		tempExp.SetSize(0);
	}
}

bool CDnnModelWrapper::classify( CClassificationResult& result ) const
{
	Dnn.RunOnce();

	const CPtr<CDnnBlob>& resultBlob = SinkLayer->GetBlob();
	NeoAssert(resultBlob->GetObjectCount() == 1);

	result.ExceptionProbability = CClassificationProbability(0);
	result.Probabilities.SetSize(ClassCount);

	if(ClassCount == 2) {
		// Binary classification is a special case
		NeoAssert(resultBlob->GetObjectSize() == 1);

		// Use a sigmoid function to estimate probabilities
		const double zeroClassProb = 1 / (1 + exp(resultBlob->GetData().GetValue()));
		result.Probabilities[0] = CClassificationProbability(zeroClassProb);
		result.Probabilities[1] = CClassificationProbability(1 - zeroClassProb);
		result.PreferredClass = zeroClassProb >= 0.5 ? 0 : 1;
	} else {
		NeoAssert(resultBlob->GetObjectSize() == ClassCount);

		tempExp.SetSize(ClassCount);
		resultBlob->CopyTo(tempExp.GetPtr(), tempExp.Size());

		// Normalize the weights before taking expf
		float maxWeight = tempExp[0];
		result.PreferredClass = 0;
		for( int i = 1; i < ClassCount; ++i ) {
			if(tempExp[i] > maxWeight) {
				maxWeight = tempExp[i];
				result.PreferredClass = i;
			}
		}

		// Use softmax to estimate probabilities
		float expSum = 0;

		for(int i = 0; i < ClassCount; ++i) {
			// To divide all exponent terms by the maximum, subtract it beforehand
			tempExp[i] = expf(tempExp[i] - maxWeight);
			expSum += tempExp[i];
		}

		for(int i = 0; i < ClassCount; ++i) {
			result.Probabilities[i] = CClassificationProbability(tempExp[i] / expSum);
		}
	}

	return true;
}

//---------------------------------------------------------------------------------------------------------------------

CPtr<IModel> CDnnTrainingModelWrapper::Train(const IProblem& trainingClassificationData)
{
	CPtr<CDnnModelWrapper> model = FINE_DEBUG_NEW CDnnModelWrapper( mathEngine );
	CPtr<CProblemSourceLayer> problem = FINE_DEBUG_NEW CProblemSourceLayer( mathEngine );

	problem->SetName(model->SourceLayer->GetName());
	problem->SetProblem(&trainingClassificationData);

	model->ClassCount = trainingClassificationData.GetClassCount();

	BuildAndTrainDnn(model->Dnn, problem, model->SourceLayer, model->SinkLayer);

	model->SourceBlob = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, 1, trainingClassificationData.GetFeatureCount() );
	model->SourceLayer->SetBlob(model->SourceBlob);

	return model.Ptr();
}

} //namespace NeoML
