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

namespace NeoML {

void CCompositeSourceLayer::SetBlobDesc( const CBlobDesc& _desc )
{
	bool isReshapeNeeded = desc.GetDataType() == CT_Invalid
		|| !desc.HasEqualDimensions( _desc )
		|| desc.GetDataType() != _desc.GetDataType();

	desc = _desc;

	if( isReshapeNeeded ) {
		ForceReshape();
		if( !outputBlobs.IsEmpty() ) {
			outputBlobs[0] = 0;
		}
	}
}

void CCompositeSourceLayer::SetBlob(CDnnBlob* _blob)
{
	NeoPresume( _blob == 0 || _blob->GetDesc().HasEqualDimensions( desc )
		|| ( _blob->GetParent() != 0 && _blob->GetParent()->GetDesc().HasEqualDimensions( desc ) ) );
	if( blob.Ptr() == _blob ) {
		return;
	}

	blob = _blob;
	if( !outputBlobs.IsEmpty() ) {
		outputBlobs[0] = 0;
	}
}

void CCompositeSourceLayer::SetDiffBlob(CDnnBlob* blob)
{ 
	if( GetDnn()->IsRecurrentMode() && blob->GetBatchLength() > 1 ) {
		diffBlob = CDnnBlob::CreateWindowBlob( blob );
	} else {
		diffBlob = blob;
	}
}

void CCompositeSourceLayer::Reshape()
{
	NeoPresume( outputDescs.Size() == 1 );
	NeoPresume( desc.GetDataType() != CT_Invalid );

	outputDescs[0] = desc;
}

void CCompositeSourceLayer::RunOnce()
{
}

void CCompositeSourceLayer::BackwardOnce()
{
	NeoAssert(outputDiffBlobs[0]->HasEqualDimensions(diffBlob));

	if( diffBlob->GetParent() != 0 ) {
		diffBlob->SetParentPos(GetDnn()->GetCurrentSequencePos() % diffBlob->GetParent()->GetBatchLength());
	}
	diffBlob->Add(outputDiffBlobs[0]);
}

void CCompositeSourceLayer::AllocateOutputBlobs()
{
	NeoPresume( outputBlobs.Size() == 1 );
	NeoPresume( blob != 0 );

	if( outputBlobs[0].Ptr() != blob.Ptr() ) {
		outputBlobs[0] = blob;
	}
}

static const int CompositeSourceLayerVersion = 2000;

void CCompositeSourceLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CompositeSourceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

////////////////////////////////////////////////////////////////////////////////

void CCompositeSinkLayer::Reshape()
{
	blob = 0;
	parentBlob = 0;
}

void CCompositeSinkLayer::RunOnce()
{
	if( blob == 0 || !GetDnn()->IsRecurrentMode() || GetDnn()->IsFirstSequencePos() ) {
		blob = inputBlobs[0];
		parentBlob = inputBlobs[0]->GetParent();
		return;
	}

	if( parentBlob == inputBlobs[0]->GetParent() ) {
		blob = inputBlobs[0];
		return;
	}

	blob->SetParentPos( inputBlobs[0]->GetParentPos() );
	blob->CopyFrom( inputBlobs[0] );
}

void CCompositeSinkLayer::SetDiffBlob(CDnnBlob* blob)
{ 
	if( GetDnn()->IsRecurrentMode() && blob->GetBatchLength() > 1 ) {
		diffBlob = CDnnBlob::CreateWindowBlob( blob );
	} else {
		diffBlob = blob;
	}
}

void CCompositeSinkLayer::BackwardOnce()
{
	NeoAssert(inputDiffBlobs[0]->HasEqualDimensions(diffBlob));
	if( diffBlob->GetParent() != 0 ) {
		diffBlob->SetParentPos(GetDnn()->GetCurrentSequencePos() % diffBlob->GetParent()->GetBatchLength());
	}
	inputDiffBlobs[0] = diffBlob;
}

static const int CompositeSinkLayerVersion = 2000;

void CCompositeSinkLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CompositeSinkLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

///////////////////////////////////////////////////////////////////////////////////////////

CCompositeLayer::CCompositeLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, name == nullptr ? "CCnnCompositeLayer" : name, true ),
	internalDnn( 0 ),
	areInternalLogsEnabled( true )
{
}

CCompositeLayer::~CCompositeLayer()
{
	if(internalDnn != 0) {
		delete internalDnn;
	}
	for( int i = layers.Size() - 1; i >= 0; i-- ) {
		CPtr<CBaseLayer> layer = layers[i];
		DeleteLayer(*layer);
	}
}

void CCompositeLayer::GetLayerList(CArray<const char*>& layerList) const
{
	layerList.SetSize(layers.Size());

	for(int i = 0; i < layers.Size(); ++i) {
		layerList[i] = layers[i]->GetName();
	}

	// Delete the back links
	for(int i = layers.Size() - 1; i >= 0; --i) {
		if(dynamic_cast<CBackLinkLayer*>(layers[i].Ptr()) != 0) {
			layerList.DeleteAt(i);
		}
	}
}

CPtr<CBaseLayer> CCompositeLayer::GetLayer(const char* name)
{
	CheckArchitecture( layerMap.Has(name), name, "layer is not in this composite layer" );
	return layerMap.Get(name);
}

CPtr<const CBaseLayer> CCompositeLayer::GetLayer(const char* name) const
{
	CheckArchitecture( layerMap.Has(name), name, "layer is not in this composite layer" );
	return layerMap.Get(name);
}

bool CCompositeLayer::HasLayer(const char* name) const
{
	return layerMap.Has(name);
}

void CCompositeLayer::AddLayerImpl(CBaseLayer& layer)
{
	CheckArchitecture( !layerMap.Has(layer.GetName()), layer.GetName(), "Layer already in this composite layer" );

	// Add the layer
	layerMap.Add(layer.GetName(), &layer);
	layers.Add(&layer);

	if(internalDnn != 0) {
		internalDnn->AddLayer(layer);
	}
}

void CCompositeLayer::DeleteLayerImpl(CBaseLayer& layer)
{
	if(internalDnn != 0) {
		internalDnn->DeleteLayer(layer);
	}

	layerMap.Delete(layer.GetName());

	for(int i = 0; i < layers.Size(); ++i) {
		if(layers[i] == &layer) {
			layers.DeleteAt(i);
			break;
		}
	}
}

CString CCompositeLayer::getSourceName(int num) const
{
	return "CompositeSource." + Str(num);
}

CString CCompositeLayer::getSinkName(int num) const
{
	return "CompositeSink." + Str(num);
}

// Creates composite layer inputs
void CCompositeLayer::createSources()
{
	int count = GetInputCount();
	while(count < sources.Size()) {
		internalDnn->DeleteLayer(*sources.Last());
		sources.DeleteAt(sources.Size() - 1);
	}
	while(count > sources.Size()) {
		auto newSource = FINE_DEBUG_NEW CCompositeSourceLayer( MathEngine() );
		newSource->SetName(getSourceName(sources.Size()));
		sources.Add(newSource);
		internalDnn->AddLayer(*newSource);
		// Set the ForceBackward flags as needed for the composite layer
		newSource->SetBackwardForced(IsBackwardNeeded());
	}
}

// Creates composite layer outputs
void CCompositeLayer::createSinks()
{
	int count = GetOutputCount();
	CheckArchitecture( count <= outputMappings.Size(), GetName(), "composite layer has too many ouputs" );
	for( int i = 0; i < min( count, sinks.Size() ); ++i ) {
		const char* inputInfoName = sinks[i]->GetInputName( 0 );
		int inputInfoOutputNumber = sinks[i]->GetInputOutputNumber( 0 );
		if( inputInfoName != outputMappings[i].InternalLayerName
			|| inputInfoOutputNumber != outputMappings[i].InternalLayerOutput )
		{
			// After the last run, the ith outputMapping has changed; reconnect it
			sinks[i]->Connect( 0, outputMappings[i].InternalLayerName, outputMappings[i].InternalLayerOutput );
		}
	}
	while(count < sinks.Size()) {
		internalDnn->DeleteLayer(*sinks.Last());
		sinks.DeleteAt(sinks.Size() - 1);
	}
	while(sinks.Size() < count) {
		auto newSink = FINE_DEBUG_NEW CCompositeSinkLayer( MathEngine() );
		int newSinkNumber = sinks.Size();
		newSink->SetName(getSinkName(newSinkNumber));
		newSink->Connect(0, outputMappings[newSinkNumber].InternalLayerName, outputMappings[newSinkNumber].InternalLayerOutput); 
		sinks.Add(newSink);
		internalDnn->AddLayer(*newSink);
	}
}

void CCompositeLayer::DeleteAllSources()
{
	while(sources.Size() > 0) {
		internalDnn->DeleteLayer(*sources.Last());
		sources.DeleteAt(sources.Size() - 1);
	}
}

void CCompositeLayer::DeleteAllSinks()
{
	while(sinks.Size() > 0) {
		internalDnn->DeleteLayer(*sinks.Last());
		sinks.DeleteAt(sinks.Size() - 1);
	}
}

void CCompositeLayer::setInputDescs()
{
	NeoPresume( sources.Size() == GetInputCount() );
	for( int i = 0; i < sources.Size(); ++i ) {
		sources[i]->SetBlobDesc( inputDescs[i] );
	}
}

void CCompositeLayer::setOutputDescs()
{
	NeoPresume( sinks.Size() == GetOutputCount() );
	for( int i = 0; i < sinks.Size(); ++i ) {
		outputDescs[i] = sinks[i]->GetInputDesc();
	}
}

// Sets the input blobs
void CCompositeLayer::setInputBlobs()
{
	// Set the blobs for each source layer
	for(int i = 0; i < sources.Size(); ++i) {
		sources[i]->SetBlob(inputBlobs[i]);
	}
}

// Sets the output blobs
void CCompositeLayer::setOutputBlobs()
{
	// Fill in the output
	for(int i = 0; i < sinks.Size(); ++i) {
		const CPtr<CDnnBlob>& out = sinks[i]->GetInputBlob();
		if( ( GetDnn()->IsRecurrentMode() || out->GetParent() == 0 ) ) {
			// This composite layer is inside a recurrent layer
			if( outputBlobs[i].Ptr() != out.Ptr() ) {
				outputBlobs[i] = out;
			}
		} else {
			NeoPresume( out->GetParent() != 0 );
			if( outputBlobs[i].Ptr() != out->GetParent() ) {
				outputBlobs[i] = out->GetParent();
			}
		}
	}
}

void CCompositeLayer::SetOutputMapping(int outputNumber, const char* internalLayerName, int internalLayerOutput)
{
	if(outputNumber >= outputMappings.Size()) {
		outputMappings.SetSize(outputNumber + 1);
	}
	outputMappings[outputNumber].InternalLayerName = internalLayerName;
	outputMappings[outputNumber].InternalLayerOutput = internalLayerOutput;
	if(internalDnn != 0) {
		internalDnn->ForceRebuild();
	}
}

void CCompositeLayer::SetInputMapping(int inputNumber, const char* internalLayerName, int internalLayerInput)
{
	GetLayer(internalLayerName)->Connect(internalLayerInput, getSourceName(inputNumber));
}

void CCompositeLayer::SetInputMapping(int inputNumber, CBaseLayer& internalLayer, int internalLayerInput)
{
	internalLayer.Connect(internalLayerInput, getSourceName(inputNumber));
}

void CCompositeLayer::OnDnnChanged( CDnn* )
{
	if(internalDnn != 0) {
		delete internalDnn;
		internalDnn = 0;
	}
	sources.DeleteAll();
	sinks.DeleteAll();
	if(GetDnn() != 0) {
		internalDnn = FINE_DEBUG_NEW CDnn(GetDnn()->Random(), GetDnn()->GetMathEngine());

		for(int i = 0; i < layers.Size(); ++i) {
			internalDnn->AddLayer(*layers[i]);
		}
	}
}

void CCompositeLayer::FilterLayerParams( float threshold )
{
	if( internalDnn != 0 ) {
		internalDnn->FilterLayersParams( threshold );
	}
}

void CCompositeLayer::SetInternalDnnParams()
{
	NeoAssert(internalDnn != 0);

	// If the backward pass requirements have changed, call reshape
	bool forcedReshape = internalDnn->IsBackwardPerformed() != GetDnn()->IsBackwardPerformed();

	// Set the internal network parameters from the external network parameters
	internalDnn->setProcessingParams(GetDnn()->IsRecurrentMode(), GetDnn()->GetMaxSequenceLength(), 
		GetDnn()->IsReverseSequense(), GetDnn()->IsBackwardPerformed());
	internalDnn->SetLog(GetDnn()->IsLogging() && areInternalLogsEnabled ? GetDnn()->GetLog() : 0);
	internalDnn->SetLogFrequency(GetDnn()->GetLogFrequency());
	internalDnn->RequestReshape(forcedReshape);
	// Switch learning on or off
	if(IsLearningEnabled()) {
		internalDnn->EnableLearning();
	} else {
		internalDnn->DisableLearning();
	}
	internalDnn->SetInitializer( GetDnn()->GetInitializer() );
}

size_t CCompositeLayer::GetOutputBlobsSize() const
{
	size_t result = 0;
	for( int i = 0; i < internalDnn->layers.Size(); i++ ) {
		result += internalDnn->layers[i]->GetOutputBlobsSize();
	}
	return result;
}

size_t CCompositeLayer::GetTrainableParametersSize() const
{
	if( !IsLearnable() ) {
		return 0;
	}

	size_t result = 0;
	for( int i = 0; i < internalDnn->layers.Size(); i++ ) {
		result += internalDnn->layers[i]->GetTrainableParametersSize();
	}
	return result;
}

void CCompositeLayer::RestartSequence()
{
	internalDnn->RestartSequence();
}

void CCompositeLayer::Reshape()
{
	// Create the source layers
	createSources();
	// Create the sink layers
	createSinks();
	// Set the input descriptors before reshape
	setInputDescs();
	// Set the internal network parameters
	SetInternalDnnParams();
	// Perform reshape for the network
	internalDnn->reshape();
	// Get the output descriptors
	setOutputDescs();
}

// Runs the internal network forward pass as defined in children
void CCompositeLayer::RunInternalDnn()
{
	internalDnn->isReuseMemoryMode = GetDnn()->isReuseMemoryMode;
	internalDnn->runOnce(GetDnn()->GetCurrentSequencePos());
}

void CCompositeLayer::RunOnce()
{
	NeoAssert(GetDnn() != 0 && internalDnn != 0);
	NeoAssert(internalDnn->IsBackwardPerformed() == GetDnn()->IsBackwardPerformed());

	if(internalDnn->GetLog() != 0) {
		*internalDnn->GetLog() << "\n";
	}

	// Set the input blobs for each source layer
	setInputBlobs();

	// Check the inputs validity
	NeoPresume(inputBlobs.Size() == sources.Size());
	for(int i = 0; i < sources.Size(); ++i) {
		NeoPresume(inputBlobs[i]->GetOwner() == sources[i]->GetBlob()->GetOwner());
	}

	// Run the internal network
	RunInternalDnn();

	// Fill in the output
	setOutputBlobs();

	// Check the outputs validity
	NeoPresume(outputBlobs.Size() == sinks.Size());
	for(int i = 0; i < sinks.Size(); ++i) {
		NeoPresume(outputBlobs[i]->GetOwner() == sinks[i]->GetInputBlob()->GetOwner());
	}

	if( GetDnn()->isReuseMemoryMode ) {
		for( int i = 0; i < sources.Size(); ++i ) {
			sources[i]->SetBlob( 0 );
		}
		for( int i = 0; i < sinks.Size(); ++i ) {
			sinks[i]->FreeInputBlob();
		}
	}
}

// Runs the internal network backward pass as defined in children
void CCompositeLayer::RunInternalDnnBackward()
{
	internalDnn->backwardRunAndLearnOnce(GetDnn()->GetCurrentSequencePos());
}

void CCompositeLayer::processBackwardOrLearn()
{
	CDnn* externalDnn = GetDnn();
	NeoAssert( internalDnn != 0 );
	NeoAssert( internalDnn->isBackwardPerformed == externalDnn->isBackwardPerformed );

	if( IsBackwardNeeded() ) {
		// Set the input diff blobs as external blobs for the source layers
		// That will make the diffs pass from the internal network to the external
		NeoAssert(inputDiffBlobs.Size() == sources.Size());
		for(int i = 0; i < sources.Size(); ++i) {
			sources[i]->SetDiffBlob(inputDiffBlobs[i]);
		}
	}
	// Set the output diff blobs as external blobs for the sink layers
	// That will make the diffs pass from the external network to the internal
	NeoAssert(sinks.Size() == outputDiffBlobs.Size());
	for(int i = 0; i < sinks.Size(); ++i) {
		sinks[i]->SetDiffBlob(outputDiffBlobs[i]);
	}
	// Reset the learning parameters because they may have changed after the last run
	CDnnSolver* solver = externalDnn->GetSolver();
	internalDnn->SetSolver(solver);
	float oldLearningRate = solver->GetLearningRate();
	solver->SetLearningRate(oldLearningRate * GetBaseLearningRate());
	float oldRegularizationL1 = solver->GetL1Regularization();
	solver->SetL1Regularization(oldRegularizationL1 * GetBaseL1RegularizationMult());
	float oldRegularizationL2 = solver->GetL2Regularization();
	solver->SetL2Regularization(oldRegularizationL2 * GetBaseL2RegularizationMult());

	if(internalDnn->GetLog()) {
		*internalDnn->GetLog() << "\n";
	}
	// Run a backward pass for the internal network
	RunInternalDnnBackward();

	solver->SetL1Regularization(oldRegularizationL1);
	solver->SetL2Regularization(oldRegularizationL2);
	solver->SetLearningRate(oldLearningRate);

	internalDnn->SetLog(0);
}

void CCompositeLayer::BackwardOnce()
{
	processBackwardOrLearn();
}

void CCompositeLayer::LearnOnce()
{
	if(!IsBackwardPerformed()) {
		processBackwardOrLearn();
	}
}

void CCompositeLayer::serializationHook(CArchive&)
{
}

static const int CompositeLayerVersion = 2000;

void CCompositeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( CompositeLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize(archive);

	if( archive.IsStoring() ) {
		archive << layers.Size();
		for( int i = 0; i < layers.Size(); i++ ) {
			SerializeLayer( archive, MathEngine(), layers[i] );
		}
		archive << outputMappings.Size();
		for(int i = 0; i < outputMappings.Size(); i++) {
			archive << outputMappings[i].InternalLayerName;
			archive << outputMappings[i].InternalLayerOutput;
		}
		serializationHook(archive);
	} else if( archive.IsLoading() ) {
		if( internalDnn != 0 ) {
			delete internalDnn;
			internalDnn = 0;
		}
		layerMap.DeleteAll();
		layers.DeleteAll();
		sinks.DeleteAll();
		sources.DeleteAll();

		CObjectArray<CBaseLayer> tmpLayers;
		int size = 0;
		archive >> size;
		tmpLayers.SetSize( size );
		for( int i = 0; i < tmpLayers.Size(); i++ ) {
			SerializeLayer( archive, MathEngine(), tmpLayers[i] );
		}
		for(int i = 0; i < tmpLayers.Size(); ++i) {
			AddLayer(*tmpLayers[i]);
		}
		outputMappings.SetSize(0);
		int outputMappingsSize;
		archive >> outputMappingsSize;
		COutputMapping outputMapping;
		for( int i = 0; i < outputMappingsSize; i++ ) {
			archive >> outputMapping.InternalLayerName;
			archive >> outputMapping.InternalLayerOutput;
			outputMappings.Add(outputMapping);
		}
		serializationHook(archive);
		ForceReshape();
		areInternalLogsEnabled = true;
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
