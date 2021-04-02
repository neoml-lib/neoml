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

#include "common.h"
#pragma hdrstop

#include "GraphOutput.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGraphOutput::CGraphOutput( const onnx::ValueInfoProto& output ) :
	CNode( output.name(), { output.name() }, {} )
{
}

bool CGraphOutput::CanCalculateOutput( const CObjectArray<const CTensorBase>& /* inputs */ ) const
{
	// This layer's only purpose is to provide results from the network to the user
	return false;
}

void CGraphOutput::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CObjectArray<const CTensorBase>& /* outputs */, CDnn& dnn )
{
	CheckNeoOnnxSupport( inputs[0] != nullptr && !inputs[0]->IsCalculated(), "COutputNode expects user-provided input" );

	CPtr<CSinkLayer> sink = new CSinkLayer( dnn.GetMathEngine() );
	sink->SetName( Name() );

	// Converting output into Onnx order
	CPtr<const CTensorBase> input = ConvertTensor( *inputs[0], CTensorLayout( inputs[0]->DimCount() ) );
	const CLayerOutput& layerOutput = dynamic_cast<const CUserTensor*>( input.Ptr() )->LayerOutput();
	sink->Connect( 0, *layerOutput.Layer, layerOutput.OutputIndex );

	dnn.AddLayer( *sink );
}

void CGraphOutput::CalculateOutput( const CObjectArray<const CTensorBase>& /* inputs */,
	CObjectArray<const CTensorBase>& /* outputs */, IMathEngine& /* mathEngine */ )
{
	CheckNeoOnnxInternal( false, "Illegal call: CGraphOutput::CalculateOutput" );
}

} // namespace NeoOnnx
