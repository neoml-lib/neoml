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

#include "PyReorgLayer.h"

class CPyReorgLayer : public CPyLayer {
public:
	explicit CPyReorgLayer( CReorgLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetStride(int value) { Layer<CReorgLayer>()->SetStride(value); }
	int GetStride() const { return Layer<CReorgLayer>()->GetStride(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Reorg" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeReorgLayer( py::module& m )
{
	py::class_<CPyReorgLayer, CPyLayer>(m, "Reorg")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyReorgLayer( *layer.Layer<CReorgLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, int stride ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CReorgLayer> reorg = new CReorgLayer( mathEngine );
			reorg->SetStride(stride);
			reorg->SetName( FindFreeLayerName( dnn, "Reorg", name ).c_str() );
			dnn.AddLayer( *reorg );
			reorg->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPyReorgLayer( *reorg, layer1.MathEngineOwner() );
		}) )
		.def( "get_stride", &CPyReorgLayer::GetStride, py::return_value_policy::reference )
		.def( "set_stride", &CPyReorgLayer::SetStride, py::return_value_policy::reference )
	;
}