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

#include "PyEltwiseLayer.h"

class CPyEltwiseSumLayer : public CPyLayer {
public:
	explicit CPyEltwiseSumLayer( CEltwiseSumLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "EltwiseSum" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyEltwiseSubLayer : public CPyLayer {
public:
	explicit CPyEltwiseSubLayer( CEltwiseSubLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "EltwiseSub" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyEltwiseMulLayer : public CPyLayer {
public:
	explicit CPyEltwiseMulLayer( CEltwiseMulLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "EltwiseMul" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyEltwiseDivLayer : public CPyLayer {
public:
	explicit CPyEltwiseDivLayer( CEltwiseDivLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "EltwiseDiv" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyEltwiseNegMulLayer : public CPyLayer {
public:
	explicit CPyEltwiseNegMulLayer( CEltwiseNegMulLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "EltwiseNegMul" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyEltwiseMaxLayer : public CPyLayer {
public:
	explicit CPyEltwiseMaxLayer( CEltwiseMaxLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "EltwiseMax" );
		return pyConstructor( py::cast(this) );
	}
};

void InitializeEltwiseLayer( py::module& m )
{
	py::class_<CPyEltwiseSumLayer, CPyLayer>(m, "EltwiseSum")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseSumLayer( *layer.Layer<CEltwiseSumLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseSumLayer> eltwise = new CEltwiseSumLayer( mathEngine );
			eltwise->SetName( FindFreeLayerName( dnn, "EltwiseSum", name ).c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseSumLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

		py::class_<CPyEltwiseSubLayer, CPyLayer>(m, "EltwiseSub")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseSubLayer( *layer.Layer<CEltwiseSubLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseSubLayer> eltwise = new CEltwiseSubLayer( mathEngine );
			eltwise->SetName( FindFreeLayerName( dnn, "EltwiseSub", name ).c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseSubLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyEltwiseMulLayer, CPyLayer>(m, "EltwiseMul")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseMulLayer( *layer.Layer<CEltwiseMulLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseMulLayer> eltwise = new CEltwiseMulLayer( mathEngine );
			eltwise->SetName( FindFreeLayerName( dnn, "EltwiseMul", name ).c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseMulLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyEltwiseDivLayer, CPyLayer>(m, "EltwiseDiv")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseDivLayer( *layer.Layer<CEltwiseDivLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseDivLayer> eltwise = new CEltwiseDivLayer( mathEngine );
			eltwise->SetName( FindFreeLayerName( dnn, "EltwiseDiv", name ).c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseDivLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyEltwiseNegMulLayer, CPyLayer>(m, "EltwiseNegMul")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseNegMulLayer( *layer.Layer<CEltwiseNegMulLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseNegMulLayer> eltwise = new CEltwiseNegMulLayer( mathEngine );
			eltwise->SetName( FindFreeLayerName( dnn, "EltwiseNegMul", name ).c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseNegMulLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyEltwiseMaxLayer, CPyLayer>(m, "EltwiseMax")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseMaxLayer( *layer.Layer<CEltwiseMaxLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseMaxLayer> eltwise = new CEltwiseMaxLayer( mathEngine );
			eltwise->SetName( FindFreeLayerName( dnn, "EltwiseMax", name ).c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseMaxLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;
}
