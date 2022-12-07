/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/Onnx/SourceReshaper.h>

namespace NeoML {

static const int SourceReshaperVersion = 0;

void CSourceReshaper::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SourceReshaperVersion );
	CBaseReshaper::Serialize( archive );
	tensor.Serialize( archive );
}

void CSourceReshaper::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 0, GetPath(), "SourceReshaper must have no inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "SourceReshaper must have 1 output" );
	CheckArchitecture( tensor.IsInitialized(), GetPath(), "SourceReshaper with uninitialized tensor" );

	tensor.CopyTo( outputShapeTensors[0] );
}

} // namespace NeoML
