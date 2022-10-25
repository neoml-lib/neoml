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

#include "common.h"
#include <NeoOnnx/NeoOnnxImport.h>

namespace NeoOnnx {

class IOptimizer : public IObject {
public:
	explicit IOptimizer( CDnn& graph ) : graph( graph ) {}
	IOptimizer( IOptimizer&& ) = delete;
	IOptimizer( const IOptimizer& ) = delete;

	virtual void Apply() = 0;

protected:
	CDnn& graph;
	CArray<CPtr<CBaseLayer>> layersToRemove{};

	static constexpr const char* const classesOfSkipLayers[]{
		"NeoMLDnnBroadcastLayer",
		"FmlCnnTransformWithoutTransposeLayer",
		"FmlCnnTransposeLayer"
	};

	CPtr<CBaseLayer> GetAnyInputLayer( const CPtr<CBaseLayer>& currentLayer, int i, const char* const layerSkipClass = "" );

	bool IsExactLayer( const CPtr<CBaseLayer>& layer, const char* layerClass, bool justCheck = false );

	bool GetExactInputLayers( const CPtr<CBaseLayer>& currentLayer,
		CPtr<CBaseLayer>& layerBase, const char* const layerBaseClass,
		CPtr<CBaseLayer>& layerData, const char* const layerDataClass, const char* const layerSkipClass );
};

} // namespace NeoOnnx

