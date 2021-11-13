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

#include "PyUnfoldLayer.h"

class CPyUnfoldLayer : public CPyLayer {
public:
	explicit CPyUnfoldLayer( CUnfoldLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLayer( layer, mathEngineOwner ) {}

	int GetFilterHeight() const { return Layer<CUnfoldLayer>()->GetFilterHeight(); }
	void SetFilterHeight( int value ) { Layer<CUnfoldLayer>()->SetFilterHeight( value ); }
	int GetFilterWidth() const { return Layer<CUnfoldLayer>()->GetFilterWidth(); }
	void SetFilterWidth( int value ) { Layer<CUnfoldLayer>()->SetFilterWidth( value ); }
	int GetStrideHeight() const { return Layer<CUnfoldLayer>()->GetStrideHeight(); }
	void SetStrideHeight( int value ) { Layer<CUnfoldLayer>()->SetStrideHeight( value ); }
	int GetStrideWidth() const { return Layer<CUnfoldLayer>()->GetStrideWidth(); }
	void SetStrideWidth( int value ) { Layer<CUnfoldLayer>()->SetStrideWidth( value ); }
	int GetPaddingHeight() const { return Layer<CUnfoldLayer>()->GetPaddingHeight(); }
	void SetPaddingHeight( int value ) { Layer<CUnfoldLayer>()->SetPaddingHeight( value ); }
	int GetPaddingWidth() const { return Layer<CUnfoldLayer>()->GetPaddingWidth(); }
	void SetPaddingWidth( int value ) { Layer<CUnfoldLayer>()->SetPaddingWidth( value ); }
	int GetDilationHeight() const { return Layer<CUnfoldLayer>()->GetDilationHeight(); }
	void SetDilationHeight( int value ) { Layer<CUnfoldLayer>()->SetDilationHeight( value ); }
	int GetDilationWidth() const { return Layer<CUnfoldLayer>()->GetDilationWidth(); }
	void SetDilationWidth( int value ) { Layer<CUnfoldLayer>()->SetDilationWidth( value ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Unfold" );
		return pyConstructor( py::cast( this ) );
	}
};

void InitializeUnfoldLayer( py::module& m )
{
	py::class_<CPyUnfoldLayer, CPyLayer>(m, "Unfold")
		.def( py::init( []( const CPyLayer& layer )
		{
			return new CPyUnfoldLayer( *layer.Layer<CUnfoldLayer>(), layer.MathEngineOwner() );
		} ) )
		.def( py::init( []( const std::string& name, const CPyLayer& inputLayer, int outputNumber,
				int filterHeight, int filterWidth, int strideHeight, int strideWidth, int paddingHeight, int paddingWidth,
				int dilationHeight, int dilationWidth )
		{
			py::gil_scoped_release release;
			CDnn& dnn = inputLayer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CUnfoldLayer> unfold = new CUnfoldLayer( mathEngine );
			unfold->SetFilterHeight( filterHeight );
			unfold->SetFilterWidth( filterWidth );
			unfold->SetStrideHeight( strideHeight );
			unfold->SetStrideWidth( strideWidth );
			unfold->SetPaddingHeight( paddingHeight );
			unfold->SetPaddingWidth( paddingWidth );
			unfold->SetDilationHeight( dilationHeight );
			unfold->SetDilationWidth( dilationWidth );
			unfold->SetName( FindFreeLayerName( dnn, "Unfold", name ).c_str() );
			dnn.AddLayer( *unfold );
			unfold->Connect( 0, inputLayer.BaseLayer(), outputNumber );
			return new CPyUnfoldLayer( *unfold, inputLayer.MathEngineOwner() );
		} ) )
		.def( "get_filter_height", &CPyUnfoldLayer::GetFilterHeight, py::return_value_policy::reference )
		.def( "set_filter_height", &CPyUnfoldLayer::SetFilterHeight, py::return_value_policy::reference )
		.def( "get_filter_width", &CPyUnfoldLayer::GetFilterWidth, py::return_value_policy::reference )
		.def( "set_filter_width", &CPyUnfoldLayer::SetFilterWidth, py::return_value_policy::reference )
		.def( "get_stride_height", &CPyUnfoldLayer::GetStrideHeight, py::return_value_policy::reference )
		.def( "set_stride_height", &CPyUnfoldLayer::SetStrideHeight, py::return_value_policy::reference )
		.def( "get_stride_width", &CPyUnfoldLayer::GetStrideWidth, py::return_value_policy::reference )
		.def( "set_stride_width", &CPyUnfoldLayer::SetStrideWidth, py::return_value_policy::reference )
		.def( "get_padding_height", &CPyUnfoldLayer::GetPaddingHeight, py::return_value_policy::reference )
		.def( "set_padding_height", &CPyUnfoldLayer::SetPaddingHeight, py::return_value_policy::reference )
		.def( "get_padding_width", &CPyUnfoldLayer::GetPaddingWidth, py::return_value_policy::reference )
		.def( "set_padding_width", &CPyUnfoldLayer::SetPaddingWidth, py::return_value_policy::reference )
		.def( "get_dilation_height", &CPyUnfoldLayer::GetDilationHeight, py::return_value_policy::reference )
		.def( "set_dilation_height", &CPyUnfoldLayer::SetDilationHeight, py::return_value_policy::reference )
		.def( "get_dilation_width", &CPyUnfoldLayer::GetDilationWidth, py::return_value_policy::reference )
		.def( "set_dilation_width", &CPyUnfoldLayer::SetDilationWidth, py::return_value_policy::reference )
	;
}
