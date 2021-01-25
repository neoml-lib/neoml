/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "PyChannelwiseConvLayer.h"

class CPyChannelwiseConvLayer : public CPyBaseConvLayer {
public:
	explicit CPyChannelwiseConvLayer( CChannelwiseConvLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseConvLayer( layer, mathEngineOwner ) {}
};

void InitializeChannelwiseConvLayer( py::module& m )
{
	py::class_<CPyChannelwiseConvLayer, CPyBaseConvLayer>(m, "ChannelwiseConv")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyChannelwiseConvLayer( *layer.Layer<CChannelwiseConvLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int filterCount, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth, int paddingHeight, int paddingWidth, bool freeTerm )
		{
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CChannelwiseConvLayer> conv = new CChannelwiseConvLayer( mathEngine );
			conv->SetName( name == "" ? findFreeLayerName( dnn, "ChannelwiseConvLayer" ).c_str() : name.c_str() );
			conv->SetFilterCount( filterCount );
			conv->SetFilterHeight( filterHeight );
			conv->SetFilterWidth( filterWidth );
			conv->SetStrideHeight( strideHeight );
			conv->SetStrideWidth( strideWidth );
			conv->SetPaddingHeight( paddingHeight );
			conv->SetPaddingWidth( paddingWidth );

			dnn.AddLayer( *conv );
			conv->Connect( 0, layer.BaseLayer(), outputNumber );
			return CPyChannelwiseConvLayer( *conv, layer.MathEngineOwner() );
		}) )
	;
}
