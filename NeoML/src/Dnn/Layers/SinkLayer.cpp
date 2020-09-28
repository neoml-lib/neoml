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

#include <NeoML/Dnn/Layers/SinkLayer.h>

namespace NeoML {

const CPtr<CDnnBlob>& CSinkLayer::GetBlob() const 
{ 
	NeoAssert( inputBlobs.Size() > 0 ); 
	return blob;
}

void CSinkLayer::Reshape()
{
	// No action: just pass the data to the user
	CheckInputs();
	if(blob == 0 || !blob->GetDesc().HasEqualDimensions(inputDescs[0])) {
		blob = 0; // reset the link to the external blob with the results
	}
}

void CSinkLayer::RunOnce()
{
	blob = inputBlobs[0];
}

void CSinkLayer::BackwardOnce()
{
	inputDiffBlobs[0]->Clear();
}

static const int SinkLayerVersion = 2000;

void CSinkLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SinkLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	int temp = 0;
	if( archive.IsStoring() ) {
		archive << temp;
	} else if( archive.IsLoading() ) {
		archive >> temp;
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
