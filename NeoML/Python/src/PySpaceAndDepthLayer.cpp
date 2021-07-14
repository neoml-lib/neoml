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

#include "PySpaceAndDepthLayer.h"

class CPySpaceToDepthLayer : public CPyLayer {
public:
	explicit CPySpaceToDepthLayer( CSpaceToDepthLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetBlockSize( int value ) { Layer<CSpaceToDepthLayer>()->SetBlockSize( value ); }
	int GetBlockSize() const { return Layer<CSpaceToDepthLayer>()->GetBlockSize(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SpaceToDepth" );
		return pyConstructor( py::cast( this ) );
	}
};

class CPyDepthToSpaceLayer : public CPyLayer {
public:
	explicit CPyDepthToSpaceLayer( CDepthToSpaceLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetBlockSize( int value ) { Layer<CDepthToSpaceLayer>()->SetBlockSize( value ); }
	int GetBlockSize() const { return Layer<CDepthToSpaceLayer>()->GetBlockSize(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "DepthToSpace" );
		return pyConstructor( py::cast( this ) );
	}
};

void InitializeSpaceAndDepthLayer( py::module& m )
{
	py::class_<CPySpaceToDepthLayer, CPyLayer>( m, "SpaceToDepth" )
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySpaceToDepthLayer( *layer.Layer<CSpaceToDepthLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& inputLayer, int outputNumber, int blockSize )
		{
			py::gil_scoped_release release;
			CDnn& dnn = inputLayer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			
			CPtr<CSpaceToDepthLayer> spaceToDepth = new CSpaceToDepthLayer( mathEngine );
			spaceToDepth->SetName( FindFreeLayerName( dnn, "SpaceToDepth", name ).c_str() );
			dnn.AddLayer( *spaceToDepth );
			spaceToDepth->Connect( 0, inputLayer.BaseLayer(), outputNumber );
			CPySpaceToDepthLayer* result = new CPySpaceToDepthLayer( *spaceToDepth, inputLayer.MathEngineOwner() );
			result->SetBlockSize( blockSize );
			return result;
		}))
		.def( "set_block_size", &CPySpaceToDepthLayer::SetBlockSize, py::return_value_policy::reference )
		.def( "get_block_size", &CPySpaceToDepthLayer::GetBlockSize, py::return_value_policy::reference )
	;

	py::class_<CPyDepthToSpaceLayer, CPyLayer>( m, "DepthToSpace" )
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyDepthToSpaceLayer( *layer.Layer<CDepthToSpaceLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& inputLayer, int outputNumber, int blockSize )
		{
			py::gil_scoped_release release;
			CDnn& dnn = inputLayer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			
			CPtr<CDepthToSpaceLayer> depthToSpace = new CDepthToSpaceLayer( mathEngine );
			depthToSpace->SetName( FindFreeLayerName( dnn, "DepthToSpace", name ).c_str() );
			dnn.AddLayer( *depthToSpace );
			depthToSpace->Connect( 0, inputLayer.BaseLayer(), outputNumber );
			CPyDepthToSpaceLayer* result = new CPyDepthToSpaceLayer( *depthToSpace, inputLayer.MathEngineOwner() );
			result->SetBlockSize( blockSize );
			return result;
		}))
		.def( "set_block_size", &CPyDepthToSpaceLayer::SetBlockSize, py::return_value_policy::reference )
		.def( "get_block_size", &CPyDepthToSpaceLayer::GetBlockSize, py::return_value_policy::reference )
	;
}
