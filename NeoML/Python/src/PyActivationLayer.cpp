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

#include "PyActivationLayer.h"

class CPyLinearLayer : public CPyLayer {
public:
	explicit CPyLinearLayer( CLinearLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetMultiplier(float multiplier) { Layer<CLinearLayer>()->SetMultiplier( multiplier ); }
	float GetMultiplier() const { return Layer<CLinearLayer>()->GetMultiplier(); }

	void SetFreeTerm(float freeTerm) { Layer<CLinearLayer>()->SetFreeTerm( freeTerm ); }
	float GetFreeTerm() const { return Layer<CLinearLayer>()->GetFreeTerm(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Linear" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

class CPyELULayer : public CPyLayer {
public:
	explicit CPyELULayer( CELULayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetAlpha(float alpha) { Layer<CELULayer>()->SetAlpha( alpha ); }
	float GetAlpha() const { return Layer<CELULayer>()->GetAlpha(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ELU" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPyReLULayer : public CPyLayer {
public:
	explicit CPyReLULayer( CReLULayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetThreshold(float alpha) { Layer<CReLULayer>()->SetUpperThreshold( alpha ); }
	float GetThreshold() const { return Layer<CReLULayer>()->GetUpperThreshold(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "ReLU" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyLeakyReLULayer : public CPyLayer {
public:
	explicit CPyLeakyReLULayer( CLeakyReLULayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetAlpha(float alpha) { Layer<CLeakyReLULayer>()->SetAlpha( alpha ); }
	float GetAlpha() const { return Layer<CLeakyReLULayer>()->GetAlpha(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "LeakyReLU" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPyHSwishLayer : public CPyLayer {
public:
	explicit CPyHSwishLayer( CHSwishLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "HSwish" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyAbsLayer : public CPyLayer {
public:
	explicit CPyAbsLayer( CAbsLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Abs" );
		return pyConstructor( py::cast(this) );
	}
};

class CPySigmoidLayer : public CPyLayer {
public:
	explicit CPySigmoidLayer( CSigmoidLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Sigmoid" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyTanhLayer : public CPyLayer {
public:
	explicit CPyTanhLayer( CTanhLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Tanh" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyHardTanhLayer : public CPyLayer {
public:
	explicit CPyHardTanhLayer( CHardTanhLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "HardTanh" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyHardSigmoidLayer : public CPyLayer {
public:
	explicit CPyHardSigmoidLayer( CHardSigmoidLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetSlope(float slope) { Layer<CHardSigmoidLayer>()->SetSlope( slope ); }
	float GetSlope() const { return Layer<CHardSigmoidLayer>()->GetSlope(); }

	void SetBias(float bias) { Layer<CHardSigmoidLayer>()->SetBias( bias ); }
	float GetBias() const { return Layer<CHardSigmoidLayer>()->GetBias(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "HardSigmoid" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

class CPyPowerLayer : public CPyLayer {
public:
	explicit CPyPowerLayer( CPowerLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetExponent(float exponent) { Layer<CPowerLayer>()->SetExponent( exponent ); }
	float GetExponent() const { return Layer<CPowerLayer>()->GetExponent(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Power" );
		return pyConstructor( py::cast(this), 0 );
	}
};

class CPyGELULayer : public CPyLayer {
public:
	explicit CPyGELULayer(CGELULayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "GELU" );
		return pyConstructor( py::cast(this) );
	}
};

class CPyExpLayer : public CPyLayer {
public:
	explicit CPyExpLayer(CExpLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Exp" );
		return pyConstructor( py::cast( this ) );
	}
};

class CPyLogLayer : public CPyLayer {
public:
	explicit CPyLogLayer(CLogLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Log" );
		return pyConstructor( py::cast( this ) );
	}
};

class CPyErfLayer : public CPyLayer {
public:
	explicit CPyErfLayer(CErfLayer& layer, CPyMathEngineOwner& mathEngineOwner) : CPyLayer(layer, mathEngineOwner) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "Erf" );
		return pyConstructor( py::cast( this ) );
	}
};


void InitializeActivationLayer( py::module& m )
{
	py::class_<CPyLinearLayer, CPyLayer>(m, "LinearLayer")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyLinearLayer( *layer.Layer<CLinearLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, float multiplier, float freeTerm ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CLinearLayer> linear = new CLinearLayer( mathEngine );
			linear->SetMultiplier( multiplier );
			linear->SetFreeTerm( freeTerm );
			linear->SetName( FindFreeLayerName( dnn, "Linear", name ).c_str() );
			dnn.AddLayer( *linear );
			linear->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyLinearLayer( *linear, layer.MathEngineOwner() );
		}) )
		.def( "get_multiplier", &CPyLinearLayer::GetMultiplier, py::return_value_policy::reference )
		.def( "set_multiplier", &CPyLinearLayer::SetMultiplier, py::return_value_policy::reference )
		.def( "get_free_term", &CPyLinearLayer::GetFreeTerm, py::return_value_policy::reference )
		.def( "set_free_term", &CPyLinearLayer::SetFreeTerm, py::return_value_policy::reference )
	;

	py::class_<CPyELULayer, CPyLayer>(m, "ELU")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyELULayer( *layer.Layer<CELULayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, float alpha ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CELULayer> elu = new CELULayer( mathEngine );
			elu->SetAlpha( alpha );
			elu->SetName( FindFreeLayerName( dnn, "ELU", name ).c_str() );
			dnn.AddLayer( *elu );
			elu->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyELULayer( *elu, layer.MathEngineOwner() );
		}) )
		.def( "get_alpha", &CPyELULayer::GetAlpha, py::return_value_policy::reference )
		.def( "set_alpha", &CPyELULayer::SetAlpha, py::return_value_policy::reference )
	;

	py::class_<CPyReLULayer, CPyLayer>(m, "ReLU")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyReLULayer( *layer.Layer<CReLULayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, float threshold ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CReLULayer> relu = new CReLULayer( mathEngine );
			relu->SetUpperThreshold( threshold );
			relu->SetName( FindFreeLayerName( dnn, "ReLU", name ).c_str() );
			dnn.AddLayer( *relu );
			relu->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyReLULayer( *relu, layer.MathEngineOwner() );
		}) )
		.def( "get_threshold", &CPyReLULayer::GetThreshold, py::return_value_policy::reference )
		.def( "set_threshold", &CPyReLULayer::SetThreshold, py::return_value_policy::reference )
	;

	py::class_<CPyLeakyReLULayer, CPyLayer>(m, "LeakyReLU")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyLeakyReLULayer( *layer.Layer<CLeakyReLULayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, float alpha ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CLeakyReLULayer> leakyReLU = new CLeakyReLULayer( mathEngine );
			leakyReLU->SetAlpha( alpha );
			leakyReLU->SetName( FindFreeLayerName( dnn, "LeakyReLU", name ).c_str() );
			dnn.AddLayer( *leakyReLU );
			leakyReLU->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyLeakyReLULayer( *leakyReLU, layer.MathEngineOwner() );
		}) )
		.def( "get_alpha", &CPyLeakyReLULayer::GetAlpha, py::return_value_policy::reference )
		.def( "set_alpha", &CPyLeakyReLULayer::SetAlpha, py::return_value_policy::reference )
	;

	py::class_<CPyHSwishLayer, CPyLayer>(m, "HSwish")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyHSwishLayer( *layer.Layer<CHSwishLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CHSwishLayer> hswish = new CHSwishLayer( mathEngine );
			hswish->SetName( FindFreeLayerName( dnn, "HSwish", name ).c_str() );
			dnn.AddLayer( *hswish );
			hswish->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyHSwishLayer( *hswish, layer.MathEngineOwner() );
		}) )
	;

	py::class_<CPyAbsLayer, CPyLayer>(m, "Abs")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyAbsLayer( *layer.Layer<CAbsLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CAbsLayer> abs = new CAbsLayer( mathEngine );
			abs->SetName( FindFreeLayerName( dnn, "Abs", name ).c_str() );
			dnn.AddLayer( *abs );
			abs->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyAbsLayer( *abs, layer.MathEngineOwner() );
		}) )
	;

	py::class_<CPySigmoidLayer, CPyLayer>(m, "Sigmoid")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPySigmoidLayer( *layer.Layer<CSigmoidLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CSigmoidLayer> sigmoid = new CSigmoidLayer( mathEngine );
			sigmoid->SetName( FindFreeLayerName( dnn, "Sigmoid", name ).c_str() );
			dnn.AddLayer( *sigmoid );
			sigmoid->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPySigmoidLayer( *sigmoid, layer.MathEngineOwner() );
		}) )
	;

	py::class_<CPyTanhLayer, CPyLayer>(m, "Tanh")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyTanhLayer( *layer.Layer<CTanhLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CTanhLayer> tanh = new CTanhLayer( mathEngine );
			tanh->SetName( FindFreeLayerName( dnn, "Tanh", name ).c_str() );
			dnn.AddLayer( *tanh );
			tanh->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyTanhLayer( *tanh, layer.MathEngineOwner() );
		}) )
	;

	py::class_<CPyHardTanhLayer, CPyLayer>(m, "HardTanh")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyHardTanhLayer( *layer.Layer<CHardTanhLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CHardTanhLayer> hardTanh = new CHardTanhLayer( mathEngine );
			hardTanh->SetName( FindFreeLayerName( dnn, "HardTanh", name ).c_str() );
			dnn.AddLayer( *hardTanh );
			hardTanh->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyHardTanhLayer( *hardTanh, layer.MathEngineOwner() );
		}) )
	;

	py::class_<CPyHardSigmoidLayer, CPyLayer>(m, "HardSigmoid")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyHardSigmoidLayer( *layer.Layer<CHardSigmoidLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, float slope, float bias ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CHardSigmoidLayer> hardSigmoid = new CHardSigmoidLayer( mathEngine );
			hardSigmoid->SetSlope(slope);
			hardSigmoid->SetBias(bias);
			hardSigmoid->SetName( FindFreeLayerName( dnn, "HardSigmoid", name ).c_str() );
			dnn.AddLayer( *hardSigmoid );
			hardSigmoid->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyHardSigmoidLayer( *hardSigmoid, layer.MathEngineOwner() );
		}) )
		.def( "get_slope", &CPyHardSigmoidLayer::GetSlope, py::return_value_policy::reference )
		.def( "set_slope", &CPyHardSigmoidLayer::SetSlope, py::return_value_policy::reference )
		.def( "get_bias", &CPyHardSigmoidLayer::GetBias, py::return_value_policy::reference )
		.def( "set_bias", &CPyHardSigmoidLayer::SetBias, py::return_value_policy::reference )
	;

	py::class_<CPyPowerLayer, CPyLayer>(m, "Power")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyPowerLayer( *layer.Layer<CPowerLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, float exponent ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CPowerLayer> power = new CPowerLayer( mathEngine );
			power->SetExponent( exponent );
			power->SetName( FindFreeLayerName( dnn, "Power", name ).c_str() );
			dnn.AddLayer( *power );
			power->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyPowerLayer( *power, layer.MathEngineOwner() );
		}) )
		.def( "get_exponent", &CPyPowerLayer::GetExponent, py::return_value_policy::reference )
		.def( "set_exponent", &CPyPowerLayer::SetExponent, py::return_value_policy::reference )
	;

	py::class_<CPyGELULayer, CPyLayer>(m, "GELU")
		.def(py::init([](const CPyLayer& layer)
			{
				return new CPyGELULayer(*layer.Layer<CGELULayer>(), layer.MathEngineOwner());
			}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber) {
				py::gil_scoped_release release;
				CDnn& dnn = layer.Dnn();
				IMathEngine& mathEngine = dnn.GetMathEngine();
				CPtr<CGELULayer> gelu = new CGELULayer(mathEngine);
				gelu->SetName( FindFreeLayerName(dnn, "GELU", name).c_str() );
				dnn.AddLayer(*gelu);
				gelu->Connect(0, layer.BaseLayer(), outputNumber);
				return new CPyGELULayer(*gelu, layer.MathEngineOwner());
			}))
	;

	py::class_<CPyExpLayer, CPyLayer>(m, "Exp")
		.def(py::init([](const CPyLayer& layer)
			{
				return new CPyExpLayer(*layer.Layer<CExpLayer>(), layer.MathEngineOwner());
			}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber) {
				py::gil_scoped_release release;
				CDnn& dnn = layer.Dnn();
				IMathEngine& mathEngine = dnn.GetMathEngine();
				CPtr<CExpLayer> exp = new CExpLayer(mathEngine);
				exp->SetName( FindFreeLayerName(dnn, "Exp", name).c_str() );
				dnn.AddLayer(*exp);
				exp->Connect(0, layer.BaseLayer(), outputNumber);
				return new CPyExpLayer(*exp, layer.MathEngineOwner());
			}))
	;

	py::class_<CPyLogLayer, CPyLayer>(m, "Log")
		.def(py::init([](const CPyLayer& layer)
			{
				return new CPyLogLayer(*layer.Layer<CLogLayer>(), layer.MathEngineOwner());
			}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber) {
				py::gil_scoped_release release;
				CDnn& dnn = layer.Dnn();
				IMathEngine& mathEngine = dnn.GetMathEngine();
				CPtr<CLogLayer> log = new CLogLayer(mathEngine);
				log->SetName( FindFreeLayerName(dnn, "Log", name).c_str() );
				dnn.AddLayer(*log);
				log->Connect(0, layer.BaseLayer(), outputNumber);
				return new CPyLogLayer(*log, layer.MathEngineOwner());
			}))
	;

	py::class_<CPyErfLayer, CPyLayer>(m, "Erf")
		.def(py::init([](const CPyLayer& layer)
			{
				return new CPyErfLayer(*layer.Layer<CErfLayer>(), layer.MathEngineOwner());
			}))
		.def(py::init([](const std::string& name, const CPyLayer& layer, int outputNumber) {
				py::gil_scoped_release release;
				CDnn& dnn = layer.Dnn();
				IMathEngine& mathEngine = dnn.GetMathEngine();
				CPtr<CErfLayer> erf = new CErfLayer(mathEngine);
				erf->SetName( FindFreeLayerName(dnn, "Erf", name).c_str() );
				dnn.AddLayer(*erf);
				erf->Connect(0, layer.BaseLayer(), outputNumber);
				return new CPyErfLayer(*erf, layer.MathEngineOwner());
			}))
	;
}
