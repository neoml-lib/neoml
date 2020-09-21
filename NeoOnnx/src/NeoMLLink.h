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

#pragma once

#include "NeoOnnxCheck.h"

namespace NeoOnnx {

// Link to the OutputIndex'th output of the NeoML Layer
struct CNeoMLLink {
	// Constructor for onnx node output, which doesn't with any NeoML layer's output
	CNeoMLLink() : Layer( nullptr ), OutputIndex( NotFound ) {}

	// Constructor for onnx node output, matching it with layer's outputIndex'th output
	CNeoMLLink( CBaseLayer* layer, int outputIndex ) : Layer( layer ), OutputIndex( outputIndex )
	{ CheckNeoOnnxInternal( layer != nullptr, "non empty output info with layer == nullptr" ); }

	CBaseLayer* Layer; // Used NeoML layer (nullptr if there is no layer mapped with this output)
	int OutputIndex; // NeoML layer's output index, mapped with this output
};

} // namespace NeoOnnx
