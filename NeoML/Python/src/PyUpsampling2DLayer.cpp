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

#include "PyTransposeLayer.h"

class CPyUpsampling2DLayer : public CPyLayer {
public:
	explicit CPyUpsampling2DLayer( CUpsampling2DLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetHeightCopyCount(int value) { Layer<CUpsampling2DLayer>()->SetHeightCopyCount(value); }
	int GetHeightCopyCount() const { return Layer<CUpsampling2DLayer>()->GetHeightCopyCount(); }
	void SetWidthCopyCount(int value) { Layer<CUpsampling2DLayer>()->SetWidthCopyCount(value); }
	int GetWidthCopyCount() const { return Layer<CUpsampling2DLayer>()->GetWidthCopyCount(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Upsampling2D" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

void InitializeUpsampling2DLayer( py::module& m )
{
	py::class_<CPyUpsampling2DLayer, CPyLayer>(m, "Upsampling2D")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyUpsampling2DLayer( *layer.Layer<CUpsampling2DLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int heightCopyCount, int widthCopyCount )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CUpsampling2DLayer> upsampling = new CUpsampling2DLayer( mathEngine );
			upsampling->SetHeightCopyCount(heightCopyCount);
			upsampling->SetWidthCopyCount(widthCopyCount);
			upsampling->SetName( name == "" ? findFreeLayerName( dnn, "Upsampling2D" ).c_str() : name.c_str() );
			dnn.AddLayer( *upsampling );
			for( int i = 0; i < layers.size(); i++ ) {
				upsampling->Connect(i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>());
			}
			return new CPyUpsampling2DLayer( *upsampling, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_width_copy_count", &CPyUpsampling2DLayer::GetWidthCopyCount, py::return_value_policy::reference )
		.def( "set_width_copy_count", &CPyUpsampling2DLayer::SetWidthCopyCount, py::return_value_policy::reference )
		.def( "get_height_copy_count", &CPyUpsampling2DLayer::GetHeightCopyCount, py::return_value_policy::reference )
		.def( "set_height_copy_count", &CPyUpsampling2DLayer::SetHeightCopyCount, py::return_value_policy::reference )
	;
}