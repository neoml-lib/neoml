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

#include "PySubSequenceLayer.h"

class CPySubSequenceLayer : public CPyLayer {
public:
	explicit CPySubSequenceLayer( CSubSequenceLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetStartPos(int value) { Layer<CSubSequenceLayer>()->SetStartPos(value); }
	int GetStartPos() const { return Layer<CSubSequenceLayer>()->GetStartPos(); }
	void SetLength(int value) { Layer<CSubSequenceLayer>()->SetLength(value); }
	int GetLength() const { return Layer<CSubSequenceLayer>()->GetLength(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SubSequence" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeSubSequenceLayer( py::module& m )
{
	py::class_<CPySubSequenceLayer, CPyLayer>(m, "SubSequence")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySubSequenceLayer( *layer.Layer<CSubSequenceLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, int pos, int len, bool reverse ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSubSequenceLayer> sequence = new CSubSequenceLayer( mathEngine );
			if( reverse ) {
				sequence->SetReverse();
			} else {
				sequence->SetStartPos(pos);
				sequence->SetLength(len);
			}
			sequence->SetName( name == "" ? findFreeLayerName( dnn, "SubSequence" ).c_str() : name.c_str() );
			dnn.AddLayer( *sequence );
			sequence->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPySubSequenceLayer( *sequence, layer1.MathEngineOwner() );
		}) )
		.def( "get_start_pos", &CPySubSequenceLayer::GetStartPos, py::return_value_policy::reference )
		.def( "set_start_pos", &CPySubSequenceLayer::SetStartPos, py::return_value_policy::reference )
		.def( "get_length", &CPySubSequenceLayer::GetLength, py::return_value_policy::reference )
		.def( "set_length", &CPySubSequenceLayer::SetLength, py::return_value_policy::reference )
	;
}