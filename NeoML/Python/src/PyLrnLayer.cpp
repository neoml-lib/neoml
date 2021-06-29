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

#include "PyLrnLayer.h"

class CPyLrnLayer : public CPyLayer
{
public:
	explicit CPyLrnLayer( CLrnLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetWindowSize( int windowSize ) { Layer<CLrnLayer>()->SetWindowSize( windowSize ); }
	int GetWindowSize() const { return Layer<CLrnLayer>()->GetWindowSize(); }

	void SetBias( float bias ) { Layer<CLrnLayer>()->SetBias( bias ); }
	float GetBias() const { return Layer<CLrnLayer>()->GetBias(); }

	void SetAlpha( float alpha ) { Layer<CLrnLayer>()->SetAlpha( alpha ); }
	float GetAlpha() const { return Layer<CLrnLayer>()->GetAlpha(); }

	void SetBeta( float beta ) { Layer<CLrnLayer>()->SetBeta( beta ); }
	float GetBeta() const { return Layer<CLrnLayer>()->GetBeta(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Lrn" );
		return pyConstructor( py::cast( this ) );
	}
};

void InitializeLrnLayer( py::module& m )
{
	py::class_<CPyLrnLayer, CPyLayer>( m, "Lrn" )
		.def( py::init( []( const CPyLayer& layer ) {
			return new CPyLrnLayer( *layer.Layer<CLrnLayer>(), layer.MathEngineOwner() );
		} ) )
		.def( py::init( []( const std::string& name, const CPyLayer& layer, int outputNumber,
			int windowSize, float bias, float alpha, float beta ) {
				CDnn& dnn = layer.Dnn();
				IMathEngine& mathEngine = dnn.GetMathEngine();
				CPtr<CLrnLayer> lrn = new CLrnLayer( mathEngine );
				lrn->SetWindowSize( windowSize );
				lrn->SetBias( bias );
				lrn->SetAlpha( alpha );
				lrn->SetBeta( beta );
				lrn->SetName( FindFreeLayerName( dnn, "Lrn", name ).c_str() );
				dnn.AddLayer( *lrn );
				lrn->Connect( 0, layer.BaseLayer(), outputNumber );
				return new CPyLrnLayer( *lrn, layer.MathEngineOwner() );
			}
		) )
		.def( "get_window_size", &CPyLrnLayer::GetWindowSize, py::return_value_policy::reference )
		.def( "set_window_size", &CPyLrnLayer::SetWindowSize, py::return_value_policy::reference )
		.def( "get_bias", &CPyLrnLayer::GetBias, py::return_value_policy::reference )
		.def( "set_bias", &CPyLrnLayer::SetBias, py::return_value_policy::reference )
		.def( "get_alpha", &CPyLrnLayer::GetAlpha, py::return_value_policy::reference )
		.def( "set_alpha", &CPyLrnLayer::SetAlpha, py::return_value_policy::reference )
		.def( "get_beta", &CPyLrnLayer::GetBeta, py::return_value_policy::reference )
		.def( "set_beta", &CPyLrnLayer::SetBeta, py::return_value_policy::reference )
	;
}
