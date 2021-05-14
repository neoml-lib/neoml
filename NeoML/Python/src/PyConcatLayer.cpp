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

#include "PyConcatLayer.h"

class CPyConcatChannelsLayer : public CPyLayer {
public:
	explicit CPyConcatChannelsLayer( CConcatChannelsLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ConcatChannels" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyConcatWidthLayer : public CPyLayer {
public:
	explicit CPyConcatWidthLayer( CConcatWidthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ConcatWidth" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyConcatHeightLayer : public CPyLayer {
public:
	explicit CPyConcatHeightLayer( CConcatHeightLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ConcatHeight" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyConcatDepthLayer : public CPyLayer {
public:
	explicit CPyConcatDepthLayer( CConcatDepthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ConcatDepth" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyConcatBatchWidthLayer : public CPyLayer {
public:
	explicit CPyConcatBatchWidthLayer( CConcatBatchWidthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ConcatBatchWidth" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyConcatObjectLayer : public CPyLayer {
public:
	explicit CPyConcatObjectLayer( CConcatObjectLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ConcatObject" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeConcatLayer( py::module& m )
{
	py::class_<CPyConcatChannelsLayer, CPyLayer>(m, "ConcatChannels")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyConcatChannelsLayer( *layer.Layer<CConcatChannelsLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CConcatChannelsLayer> concat = new CConcatChannelsLayer( mathEngine );
			concat->SetName( FindFreeLayerName( dnn, "ConcatChannels", name ).c_str() );
			dnn.AddLayer( *concat );

			for( int i = 0; i < inputs.size(); i++ ) {
				concat->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyConcatChannelsLayer( *concat, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

	py::class_<CPyConcatWidthLayer, CPyLayer>(m, "ConcatWidth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyConcatWidthLayer( *layer.Layer<CConcatWidthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CConcatWidthLayer> concat = new CConcatWidthLayer( mathEngine );
			concat->SetName( FindFreeLayerName( dnn, "ConcatWidth", name ).c_str() );
			dnn.AddLayer( *concat );

			for( int i = 0; i < inputs.size(); i++ ) {
				concat->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyConcatWidthLayer( *concat, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

	py::class_<CPyConcatHeightLayer, CPyLayer>(m, "ConcatHeight")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyConcatHeightLayer( *layer.Layer<CConcatHeightLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CConcatHeightLayer> concat = new CConcatHeightLayer( mathEngine );
			concat->SetName( FindFreeLayerName( dnn, "ConcatHeight", name ).c_str() );
			dnn.AddLayer( *concat );

			for( int i = 0; i < inputs.size(); i++ ) {
				concat->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyConcatHeightLayer( *concat, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

	py::class_<CPyConcatDepthLayer, CPyLayer>(m, "ConcatDepth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyConcatDepthLayer( *layer.Layer<CConcatDepthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CConcatDepthLayer> concat = new CConcatDepthLayer( mathEngine );
			concat->SetName( FindFreeLayerName( dnn, "ConcatDepth", name ).c_str() );
			dnn.AddLayer( *concat );

			for( int i = 0; i < inputs.size(); i++ ) {
				concat->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyConcatDepthLayer( *concat, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

	py::class_<CPyConcatBatchWidthLayer, CPyLayer>(m, "ConcatBatchWidth")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyConcatBatchWidthLayer( *layer.Layer<CConcatBatchWidthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CConcatBatchWidthLayer> concat = new CConcatBatchWidthLayer( mathEngine );
			concat->SetName( FindFreeLayerName( dnn, "ConcatBatchWidth", name ).c_str() );
			dnn.AddLayer( *concat );

			for( int i = 0; i < inputs.size(); i++ ) {
				concat->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyConcatBatchWidthLayer( *concat, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

	py::class_<CPyConcatObjectLayer, CPyLayer>(m, "ConcatObject")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyConcatObjectLayer( *layer.Layer<CConcatObjectLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& inputs, const py::list& input_outputs )
		{
			CDnn& dnn = inputs[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CConcatObjectLayer> concat = new CConcatObjectLayer( mathEngine );
			concat->SetName( FindFreeLayerName( dnn, "ConcatObject", name ).c_str() );
			dnn.AddLayer( *concat );

			for( int i = 0; i < inputs.size(); i++ ) {
				concat->Connect( i, inputs[i].cast<CPyLayer>().BaseLayer(), input_outputs[i].cast<int>() );
			}

			return new CPyConcatObjectLayer( *concat, inputs[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;
}
