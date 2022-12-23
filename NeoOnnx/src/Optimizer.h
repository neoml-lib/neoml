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
	explicit IOptimizer( CDnn& graph, const char* const classesOfSkipLayers[] ) : Graph( graph ), ClassesOfSkipLayers( classesOfSkipLayers ) {}
	IOptimizer( IOptimizer&& ) = delete;
	IOptimizer( const IOptimizer& ) = delete;

	virtual void Apply() = 0;

protected:
	CDnn& Graph;
	const char * const * ClassesOfSkipLayers{};

	// Operations on 'layersSelected' array
	auto GetLayerSelectedSize() const { return layersSelected.Size(); }
	CPtr<CBaseLayer> GetLayerSelected( int i ) { return layersSelected[i]; }
	void ClearLayersSelected() { layersSelected.DeleteAll(); }
	void AddToLayersSelected( const CPtr<CBaseLayer>& layer ) { layersSelected.Add( layer ); }

	// Returns the pointer to input layer of the 'currentLayer' of any class-type except 'layerSkipClass' class-type and adds to 'layersSelected'.
	// All found layers of class-type is equal to 'layerSkipClass' also be added to 'layersSelected', they would not be returned.
	CPtr<CBaseLayer> GetAnyInputLayer( const CPtr<CBaseLayer>& currentLayer, int inputNum, const char* const layerSkipClass = "" );

	// If layer is not nullptr and its class-type is equal to a required class-type 'layerClass', return true
	// and if addToLayersSelected == true also add this layer to 'layersSelected' array.
	bool IsExactLayer( const CPtr<CBaseLayer>& layer, const char* const layerClass, bool addToLayersSelected = true );

	// Check all input layers of the 'currentLayer' in the 'graph' to select 1 or 2 layers.
	// It adds to 'layersSelected' the first input layer  if its class-type equals 'layerBaseClass', returns it as 'layerBase'.
	// It adds to 'layersSelected' the layer-initializer  if its class-type equals 'layerDataClass' (and that is not ""), returns it as 'layerData'. 
	// All found layers of class-type is equal to 'layerSkipClass' also would be added to 'layersSelected', but would not be returned as in arguments.
	bool GetExactInputLayers( const CPtr<CBaseLayer>& currentLayer,
		CPtr<CBaseLayer>& layerBase, const char* const layerBaseClass,
		CPtr<CBaseLayer>& layerData, const char* const layerDataClass, const char* const layerSkipClass );

private:
	CArray<CPtr<CBaseLayer>> layersSelected{};
};

} // namespace NeoOnnx

