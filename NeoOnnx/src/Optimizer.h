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

	/// Returns the pointer to input layer of the 'currentLayer' of any class-type, but if this input layer's class-type is 'layerSkipClass'
	///     then get the input layer of this input layer (so go through the layers with the class-type of 'layerSkipClass')
	/// \param[in] currentLayer    the pointer to a layer in 'graph'
	/// \param[in] i               the number of input of the 'currentLayer'
	/// \param[in] layerSkipClass  the class-type string of a layer to skip, if it is not ""
	/// \returns  the pointer to a layer that has any class-type (beside layerSkipClass), else  nullptr
	CPtr<CBaseLayer> GetAnyInputLayer( const CPtr<CBaseLayer>& currentLayer, int i, const char* const layerSkipClass = "" );

	/// If layer is not nullptr and its class-type string equals to a required class-type 'layerClass' string, 
	///    add this layer to array of 'layersToRemove' (only if justCheck == false)
	/// \param[in] layer       the pointer to a layer in 'graph'
	/// \param[in] layerClass  the class-type string of a layer
	/// \param[in] justCheck   a flag to use this method only for check (not to add to  'layersToRemove')
	/// \returns  true  if layer has necessary class-type, else  false
	bool IsExactLayer( const CPtr<CBaseLayer>& layer, const char* layerClass, bool justCheck = false );

	/// Check all input layers of the 'currentLayer' in the 'graph' in search of 1 or 2 inputs:
	///    the main input 'layerBase' of given class-type ('layerBaseClass' string) 
	///    and the layer-initializer 'layerData' of given class-type ('layerDataClass' string), only if it is not "". 
	/// If the layer of class-type 'layerSkipClass' string (is not equal "") found, the input layers of this layer will also be checked,
	///    as the search is going through all layers of this class-type.
	/// \param[in]  currentLayer     the pointer to a layer in 'graph', the inputs of what to check 
	/// \param[out] layerBase        the pointer, the main input of the current layer will be stored
	/// \param[in]  layerBaseClass   the class-type string of a main input layer
	/// \param[out] layerData        the pointer, the layer-initializer of the current layer will be stored
	/// \param[in]  layerDataClass   the class-type string of a layer-initializer, if it is "", so there should be no initializer
	/// \param[in]  layerSkipClass   the class-type string of a layer to skip, if it is "", no additional skip layer-classes in this search
	/// \returns  true  if 'currentLayer' has necessary class-type inputs, else  false
	bool GetExactInputLayers( const CPtr<CBaseLayer>& currentLayer,
		CPtr<CBaseLayer>& layerBase, const char* const layerBaseClass,
		CPtr<CBaseLayer>& layerData, const char* const layerDataClass, const char* const layerSkipClass );
};

} // namespace NeoOnnx

