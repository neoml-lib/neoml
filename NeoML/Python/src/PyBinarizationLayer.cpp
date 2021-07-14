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

#include "PyBinarizationLayer.h"

class CPyEnumBinarizationLayer : public CPyLayer {
public:
	explicit CPyEnumBinarizationLayer( CEnumBinarizationLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetEnumSize(int value) { Layer<CEnumBinarizationLayer>()->SetEnumSize( value ); }
	int GetEnumSize() const { return Layer<CEnumBinarizationLayer>()->GetEnumSize(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "EnumBinarization" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPyBitSetVectorizationLayer : public CPyLayer {
public:
	explicit CPyBitSetVectorizationLayer( CBitSetVectorizationLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetBitSetSize(int value) { Layer<CBitSetVectorizationLayer>()->SetBitSetSize( value ); }
	int GetBitSetSize() const { return Layer<CBitSetVectorizationLayer>()->GetBitSetSize(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "BitSetVectorization" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeBinarizationLayer( py::module& m )
{
	py::class_<CPyEnumBinarizationLayer, CPyLayer>(m, "EnumBinarization")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyEnumBinarizationLayer( *layer.Layer<CEnumBinarizationLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int enumSize ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CEnumBinarizationLayer> bin = new CEnumBinarizationLayer( mathEngine );
			bin->SetEnumSize( enumSize );
			bin->SetName( FindFreeLayerName( dnn, "EnumBinarization", name ).c_str() );
			dnn.AddLayer( *bin );
			bin->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyEnumBinarizationLayer( *bin, layer.MathEngineOwner() );
		}) )
		.def( "get_enum_size", &CPyEnumBinarizationLayer::GetEnumSize, py::return_value_policy::reference )
		.def( "set_enum_size", &CPyEnumBinarizationLayer::SetEnumSize, py::return_value_policy::reference )
	;

	py::class_<CPyBitSetVectorizationLayer, CPyLayer>(m, "BitSetVectorization")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyBitSetVectorizationLayer( *layer.Layer<CBitSetVectorizationLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int bitsetSize ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CBitSetVectorizationLayer> bin = new CBitSetVectorizationLayer( mathEngine );
			bin->SetBitSetSize( bitsetSize );
			bin->SetName( FindFreeLayerName( dnn, "BitSetVectorization", name ).c_str() );
			dnn.AddLayer( *bin );
			bin->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyBitSetVectorizationLayer( *bin, layer.MathEngineOwner() );
		}) )
		.def( "get_bit_set_size", &CPyBitSetVectorizationLayer::GetBitSetSize, py::return_value_policy::reference )
		.def( "set_bit_set_size", &CPyBitSetVectorizationLayer::SetBitSetSize, py::return_value_policy::reference )
	;
}
