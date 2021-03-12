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

#include "PyArgmaxLayer.h"

class CPyArgmaxLayer : public CPyLayer {
public:
	explicit CPyArgmaxLayer( CArgmaxLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetDimension(int d) { Layer<CArgmaxLayer>()->SetDimension( static_cast<TBlobDim>( d ) ); }
	int GetDimension() const { return static_cast<int>( Layer<CArgmaxLayer>()->GetDimension() ); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Argmax" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeArgmaxLayer( py::module& m )
{
	py::class_<CPyArgmaxLayer, CPyLayer>(m, "Argmax")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyArgmaxLayer( *layer.Layer<CArgmaxLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int dimension ) {
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CArgmaxLayer> argmax = new CArgmaxLayer( mathEngine );
			argmax->SetDimension( static_cast<TBlobDim>(dimension) );
			argmax->SetName( name == "" ? findFreeLayerName( dnn, "ArgmaxLayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *argmax );
			argmax->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyArgmaxLayer( *argmax, layer.MathEngineOwner() );
		}) )
		.def( "get_dimension", &CPyArgmaxLayer::GetDimension, py::return_value_policy::reference )
		.def( "set_dimension", &CPyArgmaxLayer::SetDimension, py::return_value_policy::reference )
	;
}
