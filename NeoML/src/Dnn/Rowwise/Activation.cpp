/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/Rowwise/Activation.h>

namespace NeoML {

CRowwiseOperationDesc* CRowwiseActivation::GetDesc()
{
	return mathEngine.InitRowwiseActivation( desc );
}

void CRowwiseActivation::Serialize( CArchive& archive )
{
	(void) archive.SerializeVersion( 0 ); // version
	if( archive.IsStoring() ) {
		StoreActivationDesc( desc, archive );
	} else {
		desc = LoadActivationDesc( archive );
	}
}

REGISTER_NEOML_ROWWISE_OPERATION( CRowwiseActivation, "RowwiseActivationOperation" )

} // namespace NeoML
