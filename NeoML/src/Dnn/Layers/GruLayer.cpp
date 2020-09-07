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

#include <NeoML/Dnn/Layers/GruLayer.h>
#include <NeoML/Dnn/Layers/ConcatLayer.h>
#include <NeoML/Dnn/Layers/SplitLayer.h>
#include <NeoML/Dnn/Layers/ActivationLayers.h>
#include <NeoML/Dnn/Layers/EltwiseLayer.h>

namespace NeoML {

CGruLayer::CGruLayer( IMathEngine& mathEngine ) :
	CRecurrentLayer( mathEngine, "CCnnGruLayer" )
{
	buildLayer();
}

// Builds the layer
void CGruLayer::buildLayer()
{
	// Initialize the back link
	mainBackLink = FINE_DEBUG_NEW CBackLinkLayer( MathEngine() );
	AddBackLink(*mainBackLink);

	// The layers
	CPtr<CConcatObjectLayer> gateConcat = FINE_DEBUG_NEW CConcatObjectLayer( MathEngine() );
	CString gateConcatName = gateConcat->GetName() + CString(".gates");
	gateConcat->SetName(gateConcatName);
	SetInputMapping(*gateConcat);
	gateConcat->Connect(1, *mainBackLink, 0);
	AddLayer(*gateConcat);

	gateLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	CString gateLayerName = gateLayer->GetName() + CString(".gates");
	gateLayer->SetName(gateLayerName);
	gateLayer->Connect(*gateConcat);
	AddLayer(*gateLayer);

	splitLayer = FINE_DEBUG_NEW CSplitChannelsLayer( MathEngine() );
	splitLayer->SetOutputCounts2(0);
	splitLayer->Connect(*gateLayer);
	AddLayer(*splitLayer);

	CPtr<CSigmoidLayer> resetSigmoid = FINE_DEBUG_NEW CSigmoidLayer( MathEngine() );
	CString resetSigmoidName = resetSigmoid->GetName() + CString(".reset");
	resetSigmoid->SetName(resetSigmoidName);
	resetSigmoid->Connect(0, *splitLayer, G_Reset);
	AddLayer(*resetSigmoid);

	CPtr<CEltwiseMulLayer> resetGate = FINE_DEBUG_NEW CEltwiseMulLayer( MathEngine() );
	CString resetGateName = resetGate->GetName() + CString(".reset");
	resetGate->SetName(resetGateName);
	resetGate->Connect(0, *resetSigmoid);
	resetGate->Connect(1, *mainBackLink);
	AddLayer(*resetGate);

	CPtr<CConcatChannelsLayer> mainConcat = FINE_DEBUG_NEW CConcatChannelsLayer( MathEngine() );
	SetInputMapping(*mainConcat);
	mainConcat->Connect(1, *resetGate);
	AddLayer(*mainConcat);

	mainLayer = FINE_DEBUG_NEW CFullyConnectedLayer( MathEngine() );
	CString mainLayerName = mainLayer->GetName() + CString(".main");
	mainLayer->SetName(mainLayerName);
	mainLayer->Connect(*mainConcat);
	AddLayer(*mainLayer);

	CPtr<CTanhLayer> mainTanh = FINE_DEBUG_NEW CTanhLayer( MathEngine() );
	mainTanh->Connect(*mainLayer);
	AddLayer(*mainTanh);

	CPtr<CSigmoidLayer> updateSigmoid = FINE_DEBUG_NEW CSigmoidLayer( MathEngine() );
	CString updateSigmoidName = updateSigmoid->GetName() + CString(".update");
	updateSigmoid->SetName(updateSigmoidName);
	updateSigmoid->Connect(0, *splitLayer, G_Update);
	AddLayer(*updateSigmoid);

	CPtr<CEltwiseNegMulLayer> updateGate = FINE_DEBUG_NEW CEltwiseNegMulLayer( MathEngine() );
	CString updateGateName = updateGate->GetName() + CString(".update");
	updateGate->SetName(updateGateName);
	updateGate->Connect(0, *updateSigmoid);
	updateGate->Connect(1, *mainTanh);
	AddLayer(*updateGate);

	CPtr<CEltwiseMulLayer> forgetGate = FINE_DEBUG_NEW CEltwiseMulLayer( MathEngine() );
	CString forgetGateName = forgetGate->GetName() + CString(".forget");
	forgetGate->SetName(forgetGateName);
	forgetGate->Connect(0, *updateSigmoid);
	forgetGate->Connect(1, *mainBackLink);
	AddLayer(*forgetGate);

	CPtr<CEltwiseSumLayer> newState = FINE_DEBUG_NEW CEltwiseSumLayer( MathEngine() );
	newState->Connect(0, *updateGate);
	newState->Connect(1, *forgetGate);
	AddLayer(*newState);

	// Connects the back link
	mainBackLink->Connect(*newState);

	// The initial state
	SetInputMapping( 1, *mainBackLink, 1 );

	// The output
	SetOutputMapping(*newState);
}

void CGruLayer::SetHiddenSize(int size)
{
	mainLayer->SetNumberOfElements(size);
	gateLayer->SetNumberOfElements(size * G_Count);
	splitLayer->SetOutputCounts2(size);
	mainBackLink->SetDimSize(BD_Channels, size);
}

static const int GruLayerVersion = 2000;

void CGruLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( GruLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CRecurrentLayer::Serialize( archive );

	if( archive.IsLoading() ) {
		mainLayer = CheckCast<CFullyConnectedLayer>( GetLayer( mainLayer->GetName() ) );
		gateLayer = CheckCast<CFullyConnectedLayer>( GetLayer( gateLayer->GetName() ) );
		splitLayer = CheckCast<CSplitChannelsLayer>( GetLayer( splitLayer->GetName() ) );
		mainBackLink = CheckCast<CBackLinkLayer>( GetLayer( mainBackLink->GetName() ) );
	}
}

} // namespace NeoML
