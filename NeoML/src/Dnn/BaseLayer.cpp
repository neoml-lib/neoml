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

#include <NeoML/Dnn/Dnn.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Layers/CompositeLayer.h>
#include <NeoML/Dnn/Layers/BaseInPlaceLayer.h>

namespace NeoML {

CBaseLayer::CBaseLayer( IMathEngine& _mathEngine, const char* _name, bool _isLearnable ) :
	mathEngine( _mathEngine ),
	name( _name ),
	dnn( 0 ),
	isLearnable( _isLearnable ),
	isLearningEnabled( true ),
	baseLearningRate( 1 ),
	baseL2RegularizationMult( 1 ),
	baseL1RegularizationMult( 1 ),
	isBackwardNeeded( BS_Unknown ),
	isBackwardForced( false ),
	forcedReshape( true ),
	isReshapeNeeded( true ),
	lastRunNumber( 0 ),
	graphCount( 0 )
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
	NeoAssert( dnn != 0 ); // the links can be established and deleted only if the layer is in a network

	inputBlobs.DeleteAll();
	outputBlobs.DeleteAll();
	for( int cacheType = 0; cacheType < BCT_Count; ++cacheType ) {
		blobCache[cacheType].DeleteAll();
	}
	
	inputLinks.DeleteAll();
	outputs.DeleteAll();

	inputDiffBlobs.DeleteAll();
	outputDiffBlobs.DeleteAll();

	paramDiffBlobs.DeleteAll();

	readyOutputDiffs.DeleteAll();

	clearAllRuntimeBlobs();
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
			link.Layer = dnn->GetLayer(inputs[i].Name);
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

bool CBaseLayer::IsInPlaceProcessAvailable() const
{
	for(int i = 0; i < GetInputCount(); ++i) {
		const CBaseLayer* inputLayer = GetInputLayer(i);
		if(inputLayer->GetInputCount() == 0) {
			// The previous layer is a source layer so its data may not be processed in place
			// as it belongs to the user code
			return false;
		}
		if(inputLayer->outputs[inputLinks[i].OutputNumber] > 1) {
			// The previous layer output is connected to several different layer inputs
			// so its data is shared by several layers and may not be processed in place
			return false;
		}
		if(dynamic_cast<const CBaseInPlaceLayer*>(inputLayer) != 0) {
			// The previous layer is itself processing in-place, so it may be counting on not having its output blobs changed
			// So the data may not be processed in place because that would change the outputs of the previous layer
			return false;
		}
	}
	return true;
}

IMathEngine& CBaseLayer::MathEngine() const
{
	return mathEngine;
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
	CMemoryModeSwitcher switcher( MathEngine(), GetDnn()->isReuseMemoryMode );

	for( int i = 0; i < outputDescs.Size(); ++i ) {
		if( outputBlobs[i] == 0 ) {
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

CDnnBlob* CBaseLayer::switchBlobToSequentialMode(CDnnBlob* blob, TBlobCacheType cacheType, bool storeParent)
{
	if( blob == 0 || blob->GetBatchLength() == 1 ) {
		return blob;
	}

	CObjectArray<CDnnBlob>& cache = blobCache[cacheType];

	if( !storeParent ) {
		// In this case, the blob may have been left over from the last run. Looking for it
		for( int i = 0; i < cache.Size(); i++ ) {
			NeoAssert( cache[i] != blob );
			if( cache[i]->GetParent() == blob ) {
				CDnnBlob* window = cache[i];
				window->SetParentPos( dnn->GetCurrentSequencePos() % blob->GetBatchLength() );
				return window;
			}
		}
	}

	CDnnBlob* window = CDnnBlob::CreateWindowBlob(blob, 1);
	cache.Add( storeParent ? blob : window );
	window->SetParentPos(dnn->GetCurrentSequencePos() % blob->GetBatchLength());
	return window;
}

CDnnBlob* CBaseLayer::switchBlobToNonSequentialMode(CDnnBlob* blob)
{
	return blob != 0 && blob->GetParent() != 0 ? blob->GetParent() : blob;
}

void CBaseLayer::switchBlobsToSequentialMode(CObjectArray<CDnnBlob>& blobs, TBlobCacheType cacheType, bool storeParent)
{
	for(int i = 0; i < blobs.Size(); i++) {
		blobs[i] = switchBlobToSequentialMode(blobs[i], cacheType, storeParent);
	}
}

void CBaseLayer::switchBlobsToNonSequentialMode(CObjectArray<CDnnBlob>& blobs, TBlobCacheType cacheType, bool clear)
{
	for(int i = 0; i < blobs.Size(); i++) {
		blobs[i] = switchBlobToNonSequentialMode(blobs[i]);
	}
	if( clear ) {
		blobCache[cacheType].DeleteAll();
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
		|| isInPlaceProcess()
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

	MathEngine().CleanUp();

	Reshape();

	NeoPresume( inputBlobs.IsEmpty() );
	NeoPresume( outputBlobs.IsEmpty() );
	NeoPresume( outputDescs.Size() == outputs.Size() );

	inputBlobs.SetSize( inputs.Size() );
	outputBlobs.SetSize( outputs.Size() );
}

// Calls RunOnce for the layer, then recursively for its inputs
void CBaseLayer::runOnce()
{
	NeoPresume( inputBlobs.Size() == inputs.Size() );
	NeoPresume( outputBlobs.Size() == outputs.Size() );

	NeoAssert( dnn != 0 ); // possible only in a network

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
		int outputNumber = inputs[i].OutputNumber;
		CDnnBlob* prevLayerOutput = inputLayer->outputBlobs[outputNumber].Ptr();

		if( prevLayerOutput == inputBlobs[i].Ptr() ) {
			continue;
		}

		inputBlobs[i] = prevLayerOutput;

		if( GetDnn()->isReuseMemoryMode ) {
			// Notify that the output has been processed
			inputLayer->onOutputProcessed( outputNumber );
		}
	}

	AllocateOutputBlobs();

	// Create window blobs for the inputs and outputs
	if( dnn->IsRecurrentMode() ) {
		switchBlobsToSequentialMode(inputBlobs, BCT_Input, GetDnn()->isReuseMemoryMode);
		switchBlobsToSequentialMode(outputBlobs, BCT_Output, GetDnn()->isReuseMemoryMode);
		switchBlobsToSequentialMode(runtimeBlobs, BCT_Runtime, false);
		for(int i = 0; i < runtimeBlobs.Size(); i++) {
			*runtimeBlobPtrs[i] = runtimeBlobs[i];
		}
	}

	{
		RunOnce();
	}

	if( dnn->IsRecurrentMode() ) {
		switchBlobsToNonSequentialMode(inputBlobs, BCT_Input, GetDnn()->isReuseMemoryMode);
		switchBlobsToNonSequentialMode(outputBlobs, BCT_Output, GetDnn()->isReuseMemoryMode);
		switchBlobsToNonSequentialMode(runtimeBlobs, BCT_Runtime, false);
		for(int i = 0; i < runtimeBlobs.Size(); i++) {
			*runtimeBlobPtrs[i] = runtimeBlobs[i];
		}
	}

	if( GetDnn()->isReuseMemoryMode ) {
		for( int i = 0; i < inputs.Size(); ++i ) {
			inputBlobs[i] = 0;
		}

		outputProcessedCount.SetSize( outputs.Size() );
		for( int i = 0; i < outputs.Size(); ++i ) {
			outputProcessedCount[i] = 0;
		}
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

CDnnBlob* CBaseLayer::cloneBlobForDiff(CDnnBlob* blob)
{
	NeoAssert( blob != 0 );

	CDnnBlob* ret = blob->GetClone();
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

	// Check for in-place processing before processing in sequential mode
	bool isInPlace = isInPlaceProcess();

	if( dnn->IsRecurrentMode() ) {
		// Switch the input and output blobs to sequential mode (to the current position in sequence)
		switchBlobsToSequentialMode(inputBlobs, BCT_Input, false);
		switchBlobsToSequentialMode(outputBlobs, BCT_Output, false);
		switchBlobsToSequentialMode(runtimeBlobs, BCT_Runtime, false);
		for(int i = 0; i < runtimeBlobs.Size(); i++) {
			*runtimeBlobPtrs[i] = runtimeBlobs[i];
		}
	}

	// Start backward run and learning
	if( IsBackwardPerformed() ) {
		NeoAssert( inputDiffBlobs.IsEmpty() );
		// Create blobs
		for( int i = 0; i < inputBlobs.Size(); ++i ) {
			if( isInPlace ) {
				inputDiffBlobs.Add( outputDiffBlobs[i] );
			} else {
				inputDiffBlobs.Add( cloneBlobForDiff( inputBlobs[i] ) );
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
	if( dnn->IsRecurrentMode() ) {
		switchBlobsToNonSequentialMode(inputBlobs, BCT_Input, false);
		switchBlobsToNonSequentialMode(outputBlobs, BCT_Output, false);
		switchBlobsToNonSequentialMode(runtimeBlobs, BCT_Runtime, false);
		for(int i = 0; i < runtimeBlobs.Size(); i++) {
			*runtimeBlobPtrs[i] = runtimeBlobs[i];
		}
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
				outputDiffBlobs[outputNum] = cloneBlobForDiff(diffBlob);
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
	NeoAssert( newDnn == 0 || &newDnn->GetMathEngine() == &mathEngine );
	CDnn* oldDnn = dnn;
	dnn = newDnn;

	if( dnn != 0 ) {
		lastRunNumber = dnn->runNumber;
	}

	// Clear the links and blobs arrays to save memory
	inputLinks.DeleteAll();
	inputBlobs.DeleteAll();
	for( int cacheType = 0; cacheType < BCT_Count; ++cacheType ) {
		blobCache[cacheType].DeleteAll();
	}
	outputBlobs.DeleteAll();
	outputs.DeleteAll();
	outputDiffBlobs.DeleteAll();
	inputDiffBlobs.DeleteAll();
	readyOutputDiffs.DeleteAll();

	clearAllRuntimeBlobs();

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

static const int BaseLayerVersion = 2000;

void CBaseLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion(BaseLayerVersion, CDnn::ArchiveMinSupportedVersion);
	if( archive.IsStoring() ) {
		archive << name;
		archive << inputs.Size();
		for(int i = 0; i < inputs.Size(); ++i) {
			archive << inputs[i].Name;
			archive << inputs[i].OutputNumber;
		}
		archive << isBackwardForced;
		archive << isLearningEnabled;
		archive << baseLearningRate << baseL2RegularizationMult << baseL1RegularizationMult;
		SerializeBlobs( mathEngine, archive, paramBlobs );
	} else if( archive.IsLoading() ) {
		if( dnn != 0 ) {
			unlink();
		}
		archive >> name;
		int inputCount;
		archive >> inputCount;
		inputs.SetSize( inputCount );
		for(int i = 0; i < inputCount; ++i) {
			archive >> inputs[i].Name;
			archive >> inputs[i].OutputNumber;
		}
		archive >> isBackwardForced;
		archive >> isLearningEnabled;
		archive >> baseLearningRate >> baseL2RegularizationMult >> baseL1RegularizationMult;

		SerializeBlobs( mathEngine, archive, paramBlobs );
	} else {
		NeoAssert( false );
	}
}

void CBaseLayer::CheckInputs() const
{
	CheckArchitecture( !inputs.IsEmpty(), GetName(), "layer has no input" );
}

void CBaseLayer::CheckInput1() const
{
	CheckArchitecture( inputs.Size() == 1, GetName(), "layer must have exactly 1 input" );
}

void CBaseLayer::CheckOutputs() const
{
	CheckArchitecture( !outputs.IsEmpty(), GetName(), "layer has no output" );
}

void CBaseLayer::onOutputProcessed( int index )
{
	if( !GetDnn()->isReuseMemoryMode ) {
		return;
	}

	NeoPresume( outputProcessedCount.Size() > index );
	NeoPresume( outputProcessedCount[index] < outputs[index] );

	CPtr<CDnnBlob> result = outputBlobs[index];
	outputProcessedCount[index]++;
	if( outputProcessedCount[index] == outputs[index] ) {
		outputBlobs[index] = 0;
	}
}

} // namespace NeoML
