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

#include "PyPoolingLayer.h"

class CPyPoolingLayer : public CPyLayer {
public:
	explicit CPyPoolingLayer( CPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLayer( layer, mathEngineOwner ) {}

	int GetFilterHeight() const { return Layer<CPoolingLayer>()->GetFilterHeight(); }
	void SetFilterHeight( int value ) { Layer<CPoolingLayer>()->SetFilterHeight( value ); }
	int GetFilterWidth() const { return Layer<CPoolingLayer>()->GetFilterWidth(); }
	void SetFilterWidth( int value ) { Layer<CPoolingLayer>()->SetFilterWidth( value ); }

	int GetStrideHeight() const { return Layer<CPoolingLayer>()->GetStrideHeight(); }
	void SetStrideHeight( int value ) { Layer<CPoolingLayer>()->SetStrideHeight( value ); }
	int GetStrideWidth() const { return Layer<CPoolingLayer>()->GetStrideWidth(); }
	void SetStrideWidth( int value ) { Layer<CPoolingLayer>()->SetStrideWidth( value ); }
};

//------------------------------------------------------------------------------------------------------------

class CPyMaxPoolingLayer : public CPyPoolingLayer {
public:
	explicit CPyMaxPoolingLayer( CMaxPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyPoolingLayer( layer, mathEngineOwner ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyMeanPoolingLayer : public CPyPoolingLayer {
public:
	explicit CPyMeanPoolingLayer( CMeanPoolingLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyPoolingLayer( layer, mathEngineOwner ) {}
};

void InitializePoolingLayer( py::module& m )
{
	py::class_<CPyPoolingLayer, CPyLayer>(m, "Pooling")
		.def( "get_filter_height", &CPyPoolingLayer::GetFilterHeight, py::return_value_policy::reference )
		.def( "get_filter_width", &CPyPoolingLayer::GetFilterWidth, py::return_value_policy::reference )
		.def( "set_filter_height", &CPyPoolingLayer::SetFilterHeight, py::return_value_policy::reference )
		.def( "set_filter_width", &CPyPoolingLayer::SetFilterWidth, py::return_value_policy::reference )

		.def( "get_stride_height", &CPyPoolingLayer::GetStrideHeight, py::return_value_policy::reference )
		.def( "get_stride_width", &CPyPoolingLayer::GetStrideWidth, py::return_value_policy::reference )
		.def( "set_stride_height", &CPyPoolingLayer::SetStrideHeight, py::return_value_policy::reference )
		.def( "set_stride_width", &CPyPoolingLayer::SetStrideWidth, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMaxPoolingLayer, CPyPoolingLayer>(m, "MaxPooling")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyMaxPoolingLayer( *layer.Layer<CMaxPoolingLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth )
		{
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMaxPoolingLayer> pooling = new CMaxPoolingLayer( mathEngine );
			pooling->SetName( name == "" ? findFreeLayerName( dnn, "MaxPoolingLayer" ).c_str() : name.c_str() );
			pooling->SetFilterHeight( filterHeight );
			pooling->SetFilterWidth( filterWidth );
			pooling->SetStrideHeight( strideHeight );
			pooling->SetStrideWidth( strideWidth );
			dnn.AddLayer( *pooling );

			pooling->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyMaxPoolingLayer( *pooling, layer.MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMeanPoolingLayer, CPyPoolingLayer>(m, "MeanPooling")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyMeanPoolingLayer( *layer.Layer<CMeanPoolingLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int filterHeight, int filterWidth,
			int strideHeight, int strideWidth )
		{
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMeanPoolingLayer> pooling = new CMeanPoolingLayer( mathEngine );
			pooling->SetName( name == "" ? findFreeLayerName( dnn, "MeanPoolingLayer" ).c_str() : name.c_str() );
			pooling->SetFilterHeight( filterHeight );
			pooling->SetFilterWidth( filterWidth );
			pooling->SetStrideHeight( strideHeight );
			pooling->SetStrideWidth( strideWidth );
			dnn.AddLayer( *pooling );

			pooling->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyMeanPoolingLayer( *pooling, layer.MathEngineOwner() );
		}) )
	;
}
