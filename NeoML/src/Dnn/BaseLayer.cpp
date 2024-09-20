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
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>
#include <memory>

namespace NeoML {

// The maximum size of memory used for the pools
static const size_t MaxMemoryInPools = 192 * 1024 * 1024;

CBaseLayer::CBaseLayer( IMathEngine& _mathEngine, const char* _name, bool _isLearnable ) :
	mathEngine( _mathEngine ),
	name( _name ),
	isLearnable( _isLearnable )
{
}

void CBaseLayer::DisableLearning()
{
	if( !isLearningEnabled ) {
		return;
	}
	isLearningEnabled = false;
	// To recalculate isBackwardNeeded
	ForceReshape();
}

void CBaseLayer::EnableLearning()
{
	if( isLearningEnabled ) {
		return;
	}
	isLearningEnabled = true;
	// To recalculate isBackwardNeeded
	ForceReshape();
}

void CBaseLayer::SetBackwardForced( bool forced )
{
	if( forced == isBackwardForced) {
		return;
	}
	isBackwardForced = forced;
	// To recalculate isBackwardNeeded
	ForceReshape();
}

// Unlink all connections
void CBaseLayer::unlink()
{
	NeoAssert( dnn != nullptr ); // the links can be established and deleted only if the layer is in a network
	cleanUp( /*total*/true, /*linked*/false );
}

void CBaseLayer::cleanUp( bool total, bool linked )
{
	inputBlobs.DeleteAll();
	outputBlobs.DeleteAll();
	allocatedBlobs = 0;

	if( total ) {
		for( int cacheType = 0; cacheType < BCT_Count; ++cacheType ) {
			blobCache[cacheType].DeleteAll();
		}

		inputDiffBlobs.DeleteAll();
		outputDiffBlobs.DeleteAll();
		paramDiffBlobs.DeleteAll();
		readyOutputDiffs.DeleteAll();
		clearAllRuntimeBlobs();

		if( linked ) {
			ForceReshape();
		}
	}

	if( linked ) {
		inputBlobs.SetSize( inputDescs.Size() );
		outputBlobs.SetSize( outputDescs.Size() );
	} else {
		inputLinks.DeleteAll();
		outputs.DeleteAll();
		lastOutputUser.DeleteAll();
	}
}

void CBaseLayer::buildOrder()
{
	const CBaseLayer* uninitializedValue = nullptr;
	// Special value which is used when we want to disable inplace processing over specific blob
	const CBaseLayer* disabledValue = reinterpret_cast<const CBaseLayer*>( reinterpret_cast<const char*>( uninitializedValue ) - 1 );

	if( !lastOutputUser.IsEmpty() ) {
		return;
	}

	for( int i = 0; i < inputLinks.Size(); ++i ) {
		inputLinks[i].Layer->buildOrder();
	}

	const bool isSink = outputs.IsEmpty();
	for( int i = 0; i < inputLinks.Size(); ++i ) {
		const CBaseLayer*& value = inputLinks[i].Layer->lastOutputUser[inputLinks[i].OutputNumber];
		// 2 rules:
		//    1. do not overwrite disabledValue
		//    2. if we're sink then write disabledValue (in order to avoid overwriting of CDnn output blobs)
		if( value != disabledValue ) {
			value = isSink ? disabledValue : this;
		}
	}

	lastOutputUser.Add( uninitializedValue, outputs.Size() );
}

// Establish connections
void CBaseLayer::link()
{
	NeoAssert( dnn != 0 ); // the links can be established and deleted only if the layer is in a network

	isReshapeNeeded = true;	// Reshape required after any new link

	// Connect all inputs
	inputLinks.DeleteAll();
	for( int i = inputs.Size() - 1; i >= 0; i-- ) {
		if(dnn->HasLayer(inputs[i].Name)) {
			CDnnLayerLink link;
			link.OutputNumber = inputs[i].OutputNumber;
			link.Layer = dnn->getLayer(inputs[i].Name);
			inputLinks.InsertAt(link, 0);
			link.Layer->addOutput(inputs[i].OutputNumber);
		} else {
			inputs.DeleteAt(i);
		}
	}
	NeoAssert(GetInputCount() == inputLinks.Size());
}

// Link a layer to the output with the given number
void CBaseLayer::addOutput(int number)
{
	if(number + 1 > outputs.Size()) {
		outputs.Add(0, number + 1 - outputs.Size());
	}
	outputs[number] += 1;
}

void CBaseLayer::ForceReshape()
{
	forcedReshape = true;
	if( GetDnn() != 0 ) {
		GetDnn()->RequestReshape();
	}
}

void CBaseLayer::RegisterRuntimeBlob(CPtr<CDnnBlob>& blob)
{
	if(!runtimeBlobs.Has(blob)) {
		runtimeBlobs.Add(blob);
		runtimeBlobPtrs.Add(&blob);
	}
}

void CBaseLayer::clearAllRuntimeBlobs()
{
	runtimeBlobs.DeleteAll();
	runtimeBlobPtrs.DeleteAll();
}

bool CBaseLayer::InputsMayBeOverwritten() const
{
	const int inputsToOverwrite = min( GetInputCount(), GetOutputCount() );
	for(int i = 0; i < inputsToOverwrite; ++i) {
		const CBaseLayer* inputLayer = GetInputLayer(i);
		if(inputLayer->GetInputCount() == 0) {
			// The previous layer is a source layer so its data may not be processed in place
			// as it belongs to the user code
			return false;
		}
		if( inputLayer->lastOutputUser[inputLinks[i].OutputNumber] != this ) {
			// Current input will be used later by some other layer
			return false;
		}
		NeoPresume( GetDnn() != nullptr );
		if( GetDnn()->IsBackwardPerformed() && inputLayer->outputs[inputLinks[i].OutputNumber] != 1 ) {
			// Current input may be used for training other layers connected to this input
			return false;
		}
		if( ( inputLayer->blobsNeededForBackward & TOutputBlobs ) != 0 ) {
			// The previous layer needs its output for backward
			return false;
		}
		if( inputLayer->isInPlace && ( inputLayer->blobsNeededForBackward & TInputBlobs ) != 0 ) {
			// The previous layer is working inPlace and needs its input to function properly
			return false;
		}
	}
	return true;
}

// The class that switches memory reuse mode
class CMemoryModeSwitcher {
public:
	explicit CMemoryModeSwitcher( IMathEngine& _mathEngine, bool _need ) : mathEngine( _mathEngine ), need( _need )
		{ if( need ) { mathEngine.SetReuseMemoryMode( true ); } }
	~CMemoryModeSwitcher()
		{ if( need ) { mathEngine.SetReuseMemoryMode( false ); } }
public:
	IMathEngine& mathEngine;
	bool need;
};

void CBaseLayer::AllocateOutputBlobs()
{
	if( isInPlace ) {
		NeoPresume( outputBlobs.Size() <= inputBlobs.Size() );
		if( !outputBlobs.IsEmpty() && outputBlobs[0] == nullptr ) {
			for( int i = 0; i < outputBlobs.Size(); ++i ) {
				outputBlobs[i] = inputBlobs[i];
			}
		}

		return;
	}

	CMemoryModeSwitcher switcher( MathEngine(), GetDnn()->isReuseMemoryMode );

	for( int i = 0; i < outputDescs.Size(); ++i ) {
		if( outputBlobs[i] == nullptr ) {
			outputBlobs[i] = CDnnBlob::CreateBlob( MathEngine(), outputDescs[i].GetDataType(), outputDescs[i] );
		} else {
			if( !outputBlobs[i]->GetDesc().HasEqualDimensions( outputDescs[i] ) ) {
				// If this output can be connected to in-place transform. And on the second run outputBlob's shape can mismatch with outputDesc.
				// That's why now reinterpret it (because this layer can depend on outputBlob's shape).
				// After that transform will change it again.
				outputBlobs[i]->ReinterpretDimensions( outputDescs[i] );
			}
		}
	}
}

size_t CBaseLayer::GetOutputBlobsSize() const
{
	size_t result = 0;
	for( int i = 0; i < outputDescs.Size(); i++ ) {
		result += outputDescs[i].BlobSize();
	}
	return result;
}

void CBaseLayer::CleanUp( bool totalCleanUp )
{
	cleanUp( totalCleanUp, /*linked*/true );
}

size_t CBaseLayer::GetTrainableParametersSize() const
{
	if( !isLearnable ) {
		return 0;
	}

	size_t result = 0;
	for( int i = 0; i < paramBlobs.Size(); i++ ) {
		if( paramBlobs[i] != nullptr ) {
			result += paramBlobs[i]->GetDataSize();
		}
	}
	return result;
}

void CBaseLayer::transferParamsBlob( CBaseLayer& dist ) const
{
	CCompositeLayer* compositeTo = dynamic_cast<CCompositeLayer*>( &dist );
	if( compositeTo != nullptr ) {
		const CCompositeLayer* compositeFrom = CheckCast<const CCompositeLayer>( this );

		CArray<const char*> fromLayers;
		compositeFrom->GetLayerList( fromLayers );
		for( const char* layerName : fromLayers ) {
			compositeFrom->GetLayer( layerName )->transferParamsBlob( *compositeTo->GetLayer( layerName ) );
		}
	} else {
		NeoAssertMsg( dist.paramBlobs.Size() == paramBlobs.Size(), "transferParamsBlob: It isn't a copy of the layer" );
		if( IsLearnableWithEmptyParamBlobs() ) { // Special case is CTiedEmbeddingsLayer
			NeoAssert( dist.IsLearnable() && paramBlobs.Size() == 0 );
			return;
		}

		NeoAssertMsg( !dist.IsLearnable() || paramBlobs.Size() > 0,
			"transferParamsBlob: The origin dnn should be trained and reshaped to create a reference dnn" );
		// Create reference copy of dist.paramBlobs with shared buffer
		// Takes a pointer to parent's blob to access memory
		for( int j = 0; j < dist.paramBlobs.Size(); ++j ) {
			if( ContainsNullParamBlob( j ) ) {
				dist.paramBlobs[j] = nullptr; // may contain empty parameter
				continue;
			}
			NeoAssertMsg( paramBlobs[j] != nullptr, "transferParamsBlob: All trainable paramBlobs should exist" );
			dist.paramBlobs[j] = CDnnBlob::CreateWindowBlob( paramBlobs[j], paramBlobs[j]->GetDesc().BatchLength() );
		}
	}
}

void CBaseLayer::sequentialModeIfRecurrent()
{
	if( !dnn->IsRecurrentMode() ) {
		return;
	}
	// Switch the input and output blobs to sequential mode (to the current position in sequence)
	switchBlobsToSequentialMode( inputBlobs, BCT_Input, GetDnn()->isReuseMemoryMode );
	switchBlobsToSequentialMode( outputBlobs, BCT_Output, GetDnn()->isReuseMemoryMode );
	switchBlobsToSequentialMode( runtimeBlobs, BCT_Runtime, false );
	for( int i = 0; i < runtimeBlobs.Size(); i++ ) {
		*runtimeBlobPtrs[i] = runtimeBlobs[i];
	}
}

void CBaseLayer::nonSequentialModeIfRecurrent()
{
	if( !dnn->IsRecurrentMode() ) {
		return;
	}
	switchBlobsToNonSequentialMode( inputBlobs, BCT_Input, GetDnn()->isReuseMemoryMode );
	switchBlobsToNonSequentialMode( outputBlobs, BCT_Output, GetDnn()->isReuseMemoryMode );
	switchBlobsToNonSequentialMode( runtimeBlobs, BCT_Runtime, false );
	for( int i = 0; i < runtimeBlobs.Size(); i++ ) {
		*runtimeBlobPtrs[i] = runtimeBlobs[i];
	}
}

void CBaseLayer::switchBlobsToSequentialMode(CObjectArray<CDnnBlob>& blobs, TBlobCacheType cacheType, bool storeParent)
{
	CObjectArray<CDnnBlob>& cache = blobCache[cacheType];

	if( cache.Size() != blobs.Size() ) {
		cache.SetSize( blobs.Size() );
	}

	for(int i = 0; i < blobs.Size(); i++) {
		if( blobs[i] == nullptr || blobs[i]->GetBatchLength() == 1 ) {
			cache[i] = blobs[i];
			continue;
		}
		if( !storeParent && cache[i] != nullptr && cache[i]->GetParent() == blobs[i] ) {
			cache[i]->SetParentPos( dnn->GetCurrentSequencePos() % blobs[i]->GetBatchLength() );
			blobs[i] = cache[i];
			continue;
		}
		CDnnBlob* window = CDnnBlob::CreateWindowBlob(blobs[i], 1);
		window->SetParentPos( dnn->GetCurrentSequencePos() % blobs[i]->GetBatchLength() );
		cache[i] = storeParent ? blobs[i].Ptr() : window;
		blobs[i] = window;
	}
}

void CBaseLayer::switchBlobsToNonSequentialMode(CObjectArray<CDnnBlob>& blobs, TBlobCacheType cacheType, bool clear)
{
	for(int i = 0; i < blobs.Size(); i++) {
		if( blobs[i] != nullptr && blobs[i]->GetParent() != nullptr ) {
			blobs[i] = blobs[i]->GetParent();
		}
	}
	if( clear ) {
		CObjectArray<CDnnBlob>& cache = blobCache[cacheType];
		for( int i = 0; i < cache.Size(); ++i ) {
			cache[i] = nullptr;
		}
	}
}

// Sets the input blobs for the input number i
void CBaseLayer::setInputDesc(int i)
{
	inputDescs[i] = GetInputLayer( i )->outputDescs[inputLinks[i].OutputNumber];
}

// Recalculates the blobs size
void CBaseLayer::reshape()
{
	NeoAssert( dnn != 0 ); // possible only in a network

	if( !isReshapeNeeded && !forcedReshape) {
		return;
	}
	isReshapeNeeded = false;

	CArray<CBlobDesc> prevInputDescs;
	inputDescs.MoveTo( prevInputDescs );
	inputDescs.SetSize(inputs.Size());
	
	// Call the input layers reshape recursively, reset the input blobs
	for( int i = 0; i < GetInputCount(); ++i ) {
		GetInputLayer(i)->reshape();
		setInputDesc(i);
	}
	// The input blobs sizes have not changed, no need to recalculate the output size
	forcedReshape = forcedReshape
		|| inputDescs.Size() != prevInputDescs.Size()
		|| outputDescs.Size() != outputs.Size()
		|| isInPlace
		|| isComposite();

	if(!forcedReshape) {
		for(int i = 0; i < inputBlobs.Size(); i++) {
			forcedReshape = forcedReshape 
				|| !inputDescs[i].HasEqualDimensions(prevInputDescs[i]);
		}
	}
	// The inputs haven't changed, no need to reshape the outputs
	if(!forcedReshape) {
		return;
	}
	// Reshaping the layer
	forcedReshape = false;

	inputBlobs.DeleteAll();
	outputBlobs.DeleteAll();

	for( int cacheType = 0; cacheType < BCT_Count; ++cacheType ) {
		blobCache[cacheType].DeleteAll();
	}

	outputDescs.SetSize( outputs.Size() );

	inputDiffBlobs.DeleteAll();
	outputDiffBlobs.DeleteAll();
	clearAllRuntimeBlobs();
	isInPlace = false;

	if( MathEngine().GetType() == MET_Cpu
		&& GetDnn()->IsBackwardPerformed() == false
		&& MathEngine().IsDistributed() == false
		&& MathEngine().GetMemoryInPools() > MaxMemoryInPools )
	{
		MathEngine().CleanUp();
	}

	Reshape();
	blobsNeededForBackward = ( IsBackwardPerformed() ? BlobsForBackward() : 0 )
		| ( IsLearningPerformed() ? BlobsForLearn() : 0 );

	NeoPresume( inputBlobs.IsEmpty() );
	NeoPresume( outputBlobs.IsEmpty() );
	NeoPresume( outputDescs.Size() == outputs.Size() );

	inputBlobs.SetSize( inputs.Size() );
	outputBlobs.SetSize( outputs.Size() );

	runOnceCount = 0;
	runOnceTime = 0;
}

class CRunOnceTimer {
public:
	CRunOnceTimer( bool enable, IMathEngine& mathEngine, int& hitCount,
		IPerformanceCounters::CCounter::TCounterType& result );
	~CRunOnceTimer();

private:
	std::unique_ptr<IPerformanceCounters> counters;
	IPerformanceCounters::CCounter::TCounterType& result;
};

CRunOnceTimer::CRunOnceTimer( bool enable, IMathEngine& mathEngine, int& hitCount,
		IPerformanceCounters::CCounter::TCounterType& result ) :
	counters( enable ? mathEngine.CreatePerformanceCounters( true ) : nullptr ),
	result( result )
{
	if( enable ) {
		hitCount++;
		counters->Synchronise();
	}
}

CRunOnceTimer::~CRunOnceTimer()
{
	if( counters != nullptr ) {
		counters->Synchronise();
		result += ( *counters )[0].Value;
	}
}

// Calls RunOnce for the layer, then recursively for its inputs
void CBaseLayer::runOnce()
{
	NeoPresume( inputBlobs.Size() == inputs.Size() );
	NeoPresume( outputBlobs.Size() == outputs.Size() );
	NeoPresume( dnn != nullptr ); // possible only in a network

	if( lastRunNumber == dnn->runNumber ) {
		return; // has run already
	}
	lastRunNumber = dnn->runNumber;

	// Iterate through the input layers and make sure RunOnce has been called for them
	for( int i = 0; i < GetInputCount(); ++i ) {
		GetInputLayer(i)->runOnce();
	}

	// Either this is the first runOnce after reshape
	// or the input and output blobs are released directly after use
	for( int i = 0; i < inputBlobs.Size(); ++i ) {
		CBaseLayer* inputLayer = GetInputLayer( i );
		const int outputNumber = inputs[i].OutputNumber;
		CDnnBlob* prevLayerOutput = inputLayer->outputBlobs[outputNumber].Ptr();

		if( prevLayerOutput == inputBlobs[i].Ptr() ) {
			continue;
		}

		inputBlobs[i] = prevLayerOutput;
	}

	const bool mayFreeIoBlobs = GetDnn()->isReuseMemoryMode
		&& ( !GetDnn()->isBackwardPerformed || !GetDnn()->IsRecurrentMode() || GetDnn()->IsLastSequencePos()
			|| ( ( blobsNeededForBackward & TInputBlobs ) == 0 && ( !isInPlace || ( blobsNeededForBackward & TOutputBlobs ) == 0 ) ) );

	if( mayFreeIoBlobs ) {
		for( int i = 0; i < inputBlobs.Size(); ++i ) {
			CBaseLayer* inputLayer = GetInputLayer( i );
			const int outputNumber = inputs[i].OutputNumber;

			if( inputLayer->lastOutputUser[outputNumber] == this
				&& ( inputLayer->blobsNeededForBackward & TOutputBlobs ) == 0 )
			{
				inputLayer->outputBlobs[outputNumber] = nullptr;
			}
		}
	}

	AllocateOutputBlobs();
	allocatedBlobs = TInputBlobs | TOutputBlobs;

	// Create window blobs for the inputs and outputs
	sequentialModeIfRecurrent();

	{
		CRunOnceTimer timer( useTimer, MathEngine(), runOnceCount, runOnceTime );
		RunOnce();
	}

	nonSequentialModeIfRecurrent();

	if( GetDnn()->isReuseMemoryMode ) {
		setAllocatedBlobs( TOutputBlobs | blobsNeededForBackward );
	}
}

// Recalculates the isBackwardNeeded flag; recursively checks the inputs
void CBaseLayer::recheckBackwardNeeded()
{
	NeoAssert( dnn != 0 );

	if( isBackwardNeeded != BS_Unknown ) {
		return;
	}

	isBackwardNeeded = isBackwardForced ? BS_NeedsBackward : BS_DoesntNeedBackward;
	for( int i = 0; i < GetInputCount(); ++i ) {
		GetInputLayer(i)->recheckBackwardNeeded();
		if( GetInputLayer( i )->isBackwardNeeded == BS_NeedsBackward || GetInputLayer( i )->IsLearningNeeded() ) {
			isBackwardNeeded = BS_NeedsBackward;
		}
	}

	if( readyOutputDiffs.IsEmpty() && GetOutputCount() != 0 &&
		(isBackwardNeeded == BS_NeedsBackward || IsLearningNeeded()) )
	{
		readyOutputDiffs.Add( 0, GetOutputCount() );
	}
}

CDnnBlob* CBaseLayer::cloneBlobForDiff(const CBlobDesc& desc)
{
	CDnnBlob* ret = CDnnBlob::CreateBlob( MathEngine(), desc );
	ret->Clear();
	return ret;
}

//------------------------Error backpropagation and learning----------------------------------------

// Performs backward propagation and learning for the layer. Recursively calls backwardRunAndLearnOnce for its inputs
void CBaseLayer::backwardRunAndLearnOnce()
{
	// Check if all the blobs are ready
	for( int out = 0; out < readyOutputDiffs.Size(); ++out ) {
		if( readyOutputDiffs[out] < outputs[out] ) {
			return; // not enough diff blobs for the output
		}
	}
	
	sequentialModeIfRecurrent();

	// Start backward run and learning
	if( IsBackwardPerformed() ) {
		NeoAssert( inputDiffBlobs.IsEmpty() );
		// Create blobs
		for( int i = 0; i < inputBlobs.Size(); ++i ) {
			if( isInPlace && i < outputDiffBlobs.Size() ) {
				inputDiffBlobs.Add( outputDiffBlobs[i] );
			} else {
				CBlobDesc inputDiffDesc = inputDescs[i];
				if( GetDnn()->IsRecurrentMode() ) {
					inputDiffDesc.SetDimSize( BD_BatchLength, 1 );
				}
				inputDiffBlobs.Add( cloneBlobForDiff( inputDiffDesc ) );
			}
		}

		// Perform one step of error backward propagation: 
		// calculate the input error from the output one
		BackwardOnce();
	}
	// Learning: change the layer weights, using the output errors and inputs
	if( IsLearningPerformed() ) {
		if( paramDiffBlobs.Size() == 0 ) {
			// Create blobs
			for( int i = 0; i < paramBlobs.Size(); ++i ) {
				paramDiffBlobs.Add( paramBlobs[i]->GetClone() );
				paramDiffBlobs[i]->Clear();
			}
		}
		// Calculate parameter diffs
		LearnOnce();
		// Change paramBlobs layer parameters, by applying paramDiffBlobs corrections
		// according to optimizer strategy
		if( paramBlobs.Size() != 0 && ( !dnn->IsRecurrentMode() || dnn->IsFirstSequencePos() ) ) {
			GetDnn()->GetSolver()->AddDiff( this, paramDiffBlobs );
			paramDiffBlobs.DeleteAll();
		}
	}
	
	outputDiffBlobs.DeleteAll();

	if( IsBackwardPerformed() ) {
		// Pass the results further
		for( int i = 0; i < GetInputCount(); ++i ) {
			GetInputLayer(i)->transferDiffBlob( inputDiffBlobs[i], inputLinks[i].OutputNumber );
			inputDiffBlobs[i] = 0;
		}
		inputDiffBlobs.DeleteAll();

		// Recursively start backward run for the layers where all data is ready
		for( int i = 0; i < GetInputCount(); ++i ) {
			GetInputLayer(i)->backwardRunAndLearnOnce();
		}
	}

	// Set the number of ready output diffs to 0 to reset the "ready for learning" state
	for( int out = 0; out < readyOutputDiffs.Size(); ++out ) {
		readyOutputDiffs[out] = 0;
	}

	nonSequentialModeIfRecurrent();

	// If layer needs its inputs or outputs for training
	// then it needs them for all the steps of the recurrent part
	const bool freeBlobs = GetDnn()->isReuseMemoryMode
		&& ( !GetDnn()->IsRecurrentMode() || GetDnn()->IsFirstSequencePos() );
	if( freeBlobs ) {
		setAllocatedBlobs( 0 );
	}
}

// Handles the notification that output diff is ready for a given output
// If that is the last output diff necessary for learning, 
// backpropagation and learning are started for this layer
void CBaseLayer::transferDiffBlob( CDnnBlob* diffBlob, int outputNum )
{
	if( !IsBackwardPerformed() && !IsLearningPerformed() ) {
		return;	// this layer does not do backpropagation and learning
	}
	// When processing a sequence, on each iteration the diff blob has batch length equal to 1
	NeoAssert(!GetDnn()->IsRecurrentMode() || diffBlob->GetBatchLength() == 1);

	if( outputDiffBlobs.Size() == 0 ) {
		outputDiffBlobs.SetSize(outputs.Size());
	}

	if(outputs[outputNum] == 1) {
		outputDiffBlobs[outputNum] = diffBlob;
	} else {
		// If an output is connected to several inputs, create a copy of the diff blob and then add it to the others
		if(readyOutputDiffs[outputNum] == 0) {
			if( outputDiffBlobs[outputNum] == 0 ) {
				outputDiffBlobs[outputNum] = CDnnBlob::CreateBlob( MathEngine(), diffBlob->GetDesc() );
			}
			outputDiffBlobs[outputNum]->CopyFrom( diffBlob );
		} else {
			outputDiffBlobs[outputNum]->Add(diffBlob);
		}
	}

	readyOutputDiffs[outputNum] += 1;
}

//-------------------------------------------------------------------------------------------------

// Puts the layer into the newDnn network
void CBaseLayer::setDnn( CDnn* newDnn )
{
	if( newDnn == dnn ) {
		return;
	}
	NeoAssert( newDnn == nullptr || &newDnn->GetMathEngine() == &mathEngine );
	CDnn* oldDnn = dnn;
	dnn = newDnn;

	if( dnn != nullptr ) {
		lastRunNumber = dnn->runNumber;
	}
	// Clear the links and blobs arrays to save memory
	cleanUp( /*total*/true, /*linked*/false );

	OnDnnChanged( oldDnn );
}

void CBaseLayer::SetName( const char* _name )
{
	if(name == _name) {
		return;
	}
	NeoAssert(graphCount == 0);
	name = _name;
}

void CBaseLayer::Connect( int inputNumber, const char* input, int outputNumber )
{
	NeoAssert( inputNumber >= 0 );
	NeoAssert( outputNumber >= 0 );

	if( inputNumber >= inputs.Size() ) {
		inputs.SetSize( inputNumber + 1 );
	}
	inputs[inputNumber].Name = input;
	inputs[inputNumber].OutputNumber = outputNumber;

	if(dnn != 0) {
		dnn->ForceRebuild();
	}
}

void CBaseLayer::LearnOnce()
{
	NeoAssert( false );	// by default learning is disabled
}

void CBaseLayer::InitializeParamBlob(int input, CDnnBlob& blob, int inputCount)
{
	NeoAssert(GetDnn() != 0);

	if(inputCount <= 0) {
		inputCount = inputDescs[input].ObjectSize() / 2;
	}

	GetDnn()->GetInitializer()->InitializeLayerParams(blob, inputCount);
}

static constexpr int baseLayerVersion = 2000;

void CBaseLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( baseLayerVersion, CDnn::ArchiveMinSupportedVersion );

	if( archive.IsLoading() && dnn != nullptr ) {
		unlink();
	}
	archive.Serialize( name );

	int inputsSize = inputs.Size();
	archive.Serialize( inputsSize );
	inputs.SetSize( inputsSize );
	for( int i = 0; i < inputs.Size(); ++i ) {
		archive.Serialize( inputs[i].Name );
		archive.Serialize( inputs[i].OutputNumber );
	}

	archive.Serialize( isBackwardForced );
	archive.Serialize( isLearningEnabled );
	archive.Serialize( baseLearningRate );
	archive.Serialize( baseL2RegularizationMult );
	archive.Serialize( baseL1RegularizationMult );

	const bool nonReferenceDnnLayer = ( archive.IsLoading() || GetDnn() == nullptr || !GetDnn()->IsReferenceDnn() );
	if( nonReferenceDnnLayer ) {
		SerializeBlobs( mathEngine, archive, paramBlobs );
	} else { // Reference dnns will point to original dnn paramBlobs
		CObjectArray<CDnnBlob> emptyParamBlobs;
		emptyParamBlobs.SetSize( paramBlobs.Size() );
		SerializeBlobs( mathEngine, archive, emptyParamBlobs );
	}
}

void CBaseLayer::CheckInputs() const
{
	if( inputs.IsEmpty() ) {
		CheckArchitecture( false, GetPath(), "layer has no input" );
	}
}

void CBaseLayer::CheckInput1() const
{
	if( inputs.Size() != 1 ) {
		CheckArchitecture( false, GetPath(), "layer must have exactly 1 input" );
	}
}

void CBaseLayer::CheckOutputs() const
{
	if( outputs.IsEmpty() ) {
		CheckArchitecture( false, GetPath(), "layer has no output" );
	}
}

void CBaseLayer::setAllocatedBlobs( int newMask )
{
	if( ( TInputBlobs & newMask ) == 0 && ( TInputBlobs & allocatedBlobs ) != 0 ) {
		for( int i = 0; i < inputBlobs.Size(); ++i ) {
			inputBlobs[i] = nullptr;
		}
		allocatedBlobs &= ~TInputBlobs;
	}

	if( ( TOutputBlobs & newMask ) == 0 && ( TOutputBlobs & allocatedBlobs ) != 0 ) {
		for( int i = 0; i < outputBlobs.Size(); ++i ) {
			outputBlobs[i] = nullptr;
		}
		allocatedBlobs &= ~TOutputBlobs;
	}
}

} // namespace NeoML
