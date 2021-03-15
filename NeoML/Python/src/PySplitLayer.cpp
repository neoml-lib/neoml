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

#include "PySplitLayer.h"

class CPySplitChannelsLayer : public CPyLayer {
public:
	explicit CPySplitChannelsLayer( CSplitChannelsLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitChannels" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitDepthLayer : public CPyLayer {
public:
	explicit CPySplitDepthLayer( CSplitDepthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitDepth" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitWidthLayer : public CPyLayer {
public:
	explicit CPySplitWidthLayer( CSplitWidthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitWidth" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitHeightLayer : public CPyLayer {
public:
	explicit CPySplitHeightLayer( CSplitHeightLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitHeight" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitBatchWidthLayer : public CPyLayer {
public:
	explicit CPySplitBatchWidthLayer( CSplitBatchWidthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitBatchWidth" );
		return pyConstructor( py::cast(this), 0 );
	}
};


void InitializeSplitLayer( py::module& m )
{
	py::class_<CPySplitChannelsLayer, CPyLayer>(m, "SplitChannels")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitChannelsLayer( *layer.Layer<CSplitChannelsLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSplitChannelsLayer> split = new CSplitChannelsLayer( mathEngine );
			CArray<int> outputs;
			outputs.SetSize(static_cast<int>(sizes.size()));
			for( int i = 0; i < outputs.Size(); i++ ) {
				outputs[i] = reinterpret_cast<const int*>(sizes.data())[i];
			}
			split->SetOutputCounts(outputs);
			split->SetName( name == "" ? findFreeLayerName( dnn, "SplitChannels" ).c_str() : name.c_str() );
			dnn.AddLayer( *split );
			split->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPySplitChannelsLayer( *split, layer1.MathEngineOwner() );
		}) )
	;

	py::class_<CPySplitDepthLayer, CPyLayer>(m, "SplitDepth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitDepthLayer( *layer.Layer<CSplitDepthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSplitDepthLayer> split = new CSplitDepthLayer( mathEngine );
			CArray<int> outputs;
			outputs.SetSize(static_cast<int>(sizes.size()));
			for( int i = 0; i < outputs.Size(); i++ ) {
				outputs[i] = reinterpret_cast<const int*>(sizes.data())[i];
			}
			split->SetOutputCounts(outputs);
			split->SetName( name == "" ? findFreeLayerName( dnn, "SplitDepth" ).c_str() : name.c_str() );
			dnn.AddLayer( *split );
			split->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPySplitDepthLayer( *split, layer1.MathEngineOwner() );
		}) )
	;

	py::class_<CPySplitWidthLayer, CPyLayer>(m, "SplitWidth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitWidthLayer( *layer.Layer<CSplitWidthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSplitWidthLayer> split = new CSplitWidthLayer( mathEngine );
			CArray<int> outputs;
			outputs.SetSize(static_cast<int>(sizes.size()));
			for( int i = 0; i < outputs.Size(); i++ ) {
				outputs[i] = reinterpret_cast<const int*>(sizes.data())[i];
			}
			split->SetOutputCounts(outputs);
			split->SetName( name == "" ? findFreeLayerName( dnn, "SplitWidth" ).c_str() : name.c_str() );
			dnn.AddLayer( *split );
			split->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPySplitWidthLayer( *split, layer1.MathEngineOwner() );
		}) )
	;

	py::class_<CPySplitHeightLayer, CPyLayer>(m, "SplitHeight")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitHeightLayer( *layer.Layer<CSplitHeightLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSplitHeightLayer> split = new CSplitHeightLayer( mathEngine );
			CArray<int> outputs;
			outputs.SetSize(static_cast<int>(sizes.size()));
			for( int i = 0; i < outputs.Size(); i++ ) {
				outputs[i] = reinterpret_cast<const int*>(sizes.data())[i];
			}
			split->SetOutputCounts(outputs);
			split->SetName( name == "" ? findFreeLayerName( dnn, "SplitHeight" ).c_str() : name.c_str() );
			dnn.AddLayer( *split );
			split->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPySplitHeightLayer( *split, layer1.MathEngineOwner() );
		}) )
	;

	py::class_<CPySplitBatchWidthLayer, CPyLayer>(m, "SplitBatchWidth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitBatchWidthLayer( *layer.Layer<CSplitBatchWidthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			CDnn& dnn = layer1.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSplitBatchWidthLayer> split = new CSplitBatchWidthLayer( mathEngine );
			CArray<int> outputs;
			outputs.SetSize(static_cast<int>(sizes.size()));
			for( int i = 0; i < outputs.Size(); i++ ) {
				outputs[i] = reinterpret_cast<const int*>(sizes.data())[i];
			}
			split->SetOutputCounts(outputs);
			split->SetName( name == "" ? findFreeLayerName( dnn, "SplitBatchWidth" ).c_str() : name.c_str() );
			dnn.AddLayer( *split );
			split->Connect( 0, layer1.BaseLayer(), outputNumber1 );
			return new CPySplitBatchWidthLayer( *split, layer1.MathEngineOwner() );
		}) )
	;


}