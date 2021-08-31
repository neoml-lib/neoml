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

#include <NeoML/Dnn/Layers/DataLayer.h>

namespace NeoML {

void CDataLayer::SetBlob( CDnnBlob* _blob )
{
	if( _blob == blob.Ptr() ) {
		return;
	}

	blob = _blob;

	if( !outputDescs.IsEmpty() ) {
		if( blob->GetDataType() != outputDescs[0].GetDataType()
			|| !blob->GetDesc().HasEqualDimensions( outputDescs[0] ) )
		{
			outputDescs[0] = blob->GetDesc();
			ForceReshape();
		}
	}

	if( !outputBlobs.IsEmpty() ) {
		outputBlobs[0] = 0;
	}
}

void CDataLayer::Reshape()
{
	CheckOutputs();
	CheckArchitecture( GetOutputCount() == 1, GetName(), "Data layer has more than 1 output" );
	CheckArchitecture( blob.Ptr() != nullptr, GetName(), "Data layer has null data blob" );
	outputDescs[0] = blob->GetDesc();
}

void CDataLayer::RunOnce()
{
	// Just provide given blob to the network
	// No additional actions required
}

void CDataLayer::BackwardOnce()
{
	// No actions required
}

void CDataLayer::AllocateOutputBlobs()
{
	// The standard output blobs allocation does not work for us
	outputBlobs[0] = blob;
}

static const int DataLayerVersion = 0;

void CDataLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( DataLayerVersion );
	CBaseLayer::Serialize( archive );

	bool isNull = blob == nullptr;
	archive.Serialize( isNull );
	if( isNull ) {
		blob = nullptr;
	} else {
		if( archive.IsLoading() ) {
			blob = new CDnnBlob( MathEngine() );
		}
		blob->Serialize( archive );
	}
}

CDataLayer* Data( CDnn& network, const char* name )
{
	CPtr<CDataLayer> data = new CDataLayer( network.GetMathEngine() );
	data->SetName( name );
	network.AddLayer( *data );
	return data;
}

} // namespace NeoML
