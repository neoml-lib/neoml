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

#include "PyMultichannelLookupLayer.h"

class CPyMultichannelLookupLayer : public CPyLayer {
public:
	explicit CPyMultichannelLookupLayer( CMultichannelLookupLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void Initialize( const CPyInitializer& initializer )
	{
		Layer<CMultichannelLookupLayer>()->Initialize( initializer.Initializer<CDnnInitializer>() );
	}
	void Clear()
	{
		Layer<CMultichannelLookupLayer>()->Initialize( 0 );
	}

	py::list GetDimensions() const
	{
		const CArray<CLookupDimension>& res = Layer<CMultichannelLookupLayer>()->GetDimensions();

        py::list list;
		for(int i = 0; i < res.Size(); i++) {
			auto t = py::tuple(2);
			t[0] = res[i].VectorCount;
			t[1] = res[i].VectorSize;
			list.append(t);
		}
		return list;
	}
	void SetDimensions(py::list list)
	{
		CArray<CLookupDimension> dims;
		for(int i = 0; i < list.size(); i++) {
			CLookupDimension d;
			d.VectorCount = list[i].cast<py::tuple>()[0].cast<int>();
			d.VectorSize = list[i].cast<py::tuple>()[1].cast<int>();
			dims.Add(d);
		}
		Layer<CMultichannelLookupLayer>()->SetDimensions(dims);
	}

	CPyBlob GetEmbeddings(int index) const
	{
		return CPyBlob( MathEngineOwner(), const_cast<CDnnBlob*>(Layer<CMultichannelLookupLayer>()->GetEmbeddings(index)) );
	}
	void SetEmbeddings(int index, const CPyBlob& blob)
	{
		Layer<CMultichannelLookupLayer>()->SetEmbeddings( blob.Blob(), index );
	}

	bool GetUseFrameworkLearning() const { return Layer<CMultichannelLookupLayer>()->IsUseFrameworkLearning(); }
	void SetUseFrameworkLearning(bool value) { Layer<CMultichannelLookupLayer>()->SetUseFrameworkLearning(value); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MultichannelLookup" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeMultichannelLookupLayer( py::module& m )
{
	py::class_<CPyMultichannelLookupLayer, CPyLayer>(m, "MultichannelLookup")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyMultichannelLookupLayer( *layer.Layer<CMultichannelLookupLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs, const py::list& dimensions )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMultichannelLookupLayer> lookup = new CMultichannelLookupLayer( mathEngine );
			lookup->SetName( FindFreeLayerName( dnn, "MultichannelLookup", name ).c_str() );
			CArray<CLookupDimension> d;
			for( int i = 0; i < dimensions.size(); i++ ) {
				py::tuple t = dimensions[i].cast<py::tuple>();
				CLookupDimension dimension;
				dimension.VectorCount = t[0].cast<int>();
				dimension.VectorSize = t[1].cast<int>();
				d.Add( dimension );
			}
			lookup->SetDimensions( d );

			dnn.AddLayer( *lookup );

			for( int i = 0; i < inputs.size(); i++ ) {
				lookup->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyMultichannelLookupLayer( *lookup, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_dimensions", &CPyMultichannelLookupLayer::GetDimensions, py::return_value_policy::reference )
		.def( "set_dimensions", &CPyMultichannelLookupLayer::SetDimensions, py::return_value_policy::reference )
		.def( "get_embeddings", &CPyMultichannelLookupLayer::GetEmbeddings, py::return_value_policy::reference )
		.def( "set_embeddings", &CPyMultichannelLookupLayer::SetEmbeddings, py::return_value_policy::reference )
		.def( "get_framework_learning", &CPyMultichannelLookupLayer::GetUseFrameworkLearning, py::return_value_policy::reference )
		.def( "set_framework_learning", &CPyMultichannelLookupLayer::SetUseFrameworkLearning, py::return_value_policy::reference )
		.def( "initialize", &CPyMultichannelLookupLayer::Initialize, py::return_value_policy::reference )
		.def( "clear", &CPyMultichannelLookupLayer::Clear, py::return_value_policy::reference )
	;
}
