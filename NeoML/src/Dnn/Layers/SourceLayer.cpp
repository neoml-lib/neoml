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

#include <NeoML/Dnn/Layers/SourceLayer.h>

namespace NeoML {

void CSourceLayer::SetBlob( CDnnBlob* _blob )
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

void CSourceLayer::Reshape()
{
	CheckOutputs();
	CheckLayerArchitecture( GetOutputCount() == 1, "Source layer has more than 1 output" );
	CheckLayerArchitecture( blob.Ptr() != 0, "Source layer has null data blob" );
	outputDescs[0] = blob->GetDesc();
}

void CSourceLayer::RunOnce()
{
	// No action: the data will be filled by the user
}

void CSourceLayer::BackwardOnce()
{
	// No action
}

void CSourceLayer::AllocateOutputBlobs()
{
	// The standard output blobs allocation does not work for us
	outputBlobs[0] = blob;
}

static const int SourceLayerVersion = 2001;

void CSourceLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( SourceLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	// v2001 - storeBlob flag was added
	if( version >= 2001 ) {
		archive.Serialize( storeBlob );
		if( storeBlob ) {
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
	} else if( archive.IsLoading() ) {
		storeBlob = false;
	}
}

CSourceLayer* Source( CDnn& network, const char* name )
{
	CPtr<CSourceLayer> source = new CSourceLayer( network.GetMathEngine() );
	source->SetName( name );
	network.AddLayer( *source );
	return source;
}

} // namespace NeoML
