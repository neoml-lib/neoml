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

py::array CPyBaseSplitLayer::GetOutputCounts() const {
	const auto& fineCounts = Layer<CBaseSplitLayer>()->GetOutputCounts();

	py::array_t<int, py::array::c_style> counts( fineCounts.Size() );
	auto countsData = counts.mutable_unchecked<>();

	for( int i = 0; i < fineCounts.Size(); ++i ) {
		countsData( i ) = fineCounts[i];
	}

	return counts;
}

void CPyBaseSplitLayer::SetOutputCounts( py::array counts ) {
	NeoAssert( counts.ndim() == 1 );
	NeoAssert( counts.dtype().is( py::dtype::of<int>() ) );

	CArray<int> fineCounts;
	fineCounts.SetSize( static_cast<int>(counts.size()) );
	for( int i = 0; i < fineCounts.Size(); i++ ) {
		fineCounts[i] = static_cast<const int*>(counts.data())[i];
	}
	Layer<CBaseSplitLayer>()->SetOutputCounts( fineCounts );
}

class CPySplitChannelsLayer : public CPyBaseSplitLayer {
public:
	explicit CPySplitChannelsLayer( CSplitChannelsLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseSplitLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const override
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitChannels" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitDepthLayer : public CPyBaseSplitLayer {
public:
	explicit CPySplitDepthLayer( CSplitDepthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseSplitLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const override
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitDepth" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitWidthLayer : public CPyBaseSplitLayer {
public:
	explicit CPySplitWidthLayer( CSplitWidthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseSplitLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const override
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitWidth" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitHeightLayer : public CPyBaseSplitLayer {
public:
	explicit CPySplitHeightLayer( CSplitHeightLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseSplitLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const override
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitHeight" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitListSizeLayer : public CPyBaseSplitLayer {
public:
	explicit CPySplitListSizeLayer( CSplitListSizeLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseSplitLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const override
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitListSize" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitBatchWidthLayer : public CPyBaseSplitLayer {
public:
	explicit CPySplitBatchWidthLayer( CSplitBatchWidthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseSplitLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const override
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitBatchWidth" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPySplitBatchLengthLayer : public CPyBaseSplitLayer {
public:
	explicit CPySplitBatchLengthLayer( CSplitBatchLengthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyBaseSplitLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const override
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SplitBatchLength" );
		return pyConstructor( py::cast(this), 0 );
	}
};

template <class Layer, class PythonLayer>
PythonLayer* InitSplit( const std::string& className, const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes )
{
	static_assert( std::is_base_of<CPyBaseSplitLayer, PythonLayer>::value, "PySplitLayer.cpp, InitSplit" );
	static_assert( std::is_constructible<PythonLayer, Layer&, CPyMathEngineOwner&>::value, "PySplitLayer.cpp, InitSplit" );

	py::gil_scoped_release release;
	CDnn& dnn = layer1.Dnn();
	CPtr<Layer> split = new Layer( dnn.GetMathEngine() );
	split->SetName( FindFreeLayerName( dnn, className, name ).c_str() );
	dnn.AddLayer( *split );
	split->Connect( 0, layer1.BaseLayer(), outputNumber1 );
	std::unique_ptr<PythonLayer> layer( new PythonLayer( *split, layer1.MathEngineOwner() ) );
	layer->SetOutputCounts( sizes );
	return layer.release();
}

void InitializeSplitLayer( py::module& m )
{
	py::class_<CPyBaseSplitLayer, CPyLayer>(m, "BaseSplit")
		.def( "get_output_counts", &CPyBaseSplitLayer::GetOutputCounts, py::return_value_policy::move )
		.def( "set_output_counts", &CPyBaseSplitLayer::SetOutputCounts, py::return_value_policy::reference )
	;

	py::class_<CPySplitChannelsLayer, CPyBaseSplitLayer>(m, "SplitChannels")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitChannelsLayer( *layer.Layer<CSplitChannelsLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			return InitSplit<CSplitChannelsLayer, CPySplitChannelsLayer>( "SplitChannels", name, layer1, outputNumber1, sizes );
		}) )
	;

	py::class_<CPySplitDepthLayer, CPyBaseSplitLayer>(m, "SplitDepth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitDepthLayer( *layer.Layer<CSplitDepthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			return InitSplit<CSplitDepthLayer, CPySplitDepthLayer>( "SplitDepth", name, layer1, outputNumber1, sizes );
		}) )
	;

	py::class_<CPySplitWidthLayer, CPyBaseSplitLayer>(m, "SplitWidth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitWidthLayer( *layer.Layer<CSplitWidthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			return InitSplit<CSplitWidthLayer, CPySplitWidthLayer>( "SplitWidth", name, layer1, outputNumber1, sizes );
		}) )
	;

	py::class_<CPySplitHeightLayer, CPyBaseSplitLayer>(m, "SplitHeight")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitHeightLayer( *layer.Layer<CSplitHeightLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			return InitSplit<CSplitHeightLayer, CPySplitHeightLayer>( "SplitHeight", name, layer1, outputNumber1, sizes );
		}) )
	;

	py::class_<CPySplitListSizeLayer, CPyBaseSplitLayer>(m, "SplitListSize")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitListSizeLayer( *layer.Layer<CSplitListSizeLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			return InitSplit<CSplitListSizeLayer, CPySplitListSizeLayer>( "SplitListSize", name, layer1, outputNumber1, sizes );
		}) )
	;

	py::class_<CPySplitBatchWidthLayer, CPyBaseSplitLayer>(m, "SplitBatchWidth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitBatchWidthLayer( *layer.Layer<CSplitBatchWidthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			return InitSplit<CSplitBatchWidthLayer, CPySplitBatchWidthLayer>( "SplitBatchWidth", name, layer1, outputNumber1, sizes );
		}) )
	;

	py::class_<CPySplitBatchLengthLayer, CPyBaseSplitLayer>(m, "SplitBatchLength")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySplitBatchLengthLayer( *layer.Layer<CSplitBatchLengthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer1, int outputNumber1, py::array sizes ) {
			return InitSplit<CSplitBatchLengthLayer, CPySplitBatchLengthLayer>( "SplitBatchLength", name, layer1, outputNumber1, sizes );
		}) )
	;
}
