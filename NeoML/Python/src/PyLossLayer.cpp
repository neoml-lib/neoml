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

#include "PyLossLayer.h"
#include "PyDnnBlob.h"

class CPyCrossEntropyLossLayer : public CPyLossLayer {
public:
	explicit CPyCrossEntropyLossLayer( CCrossEntropyLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	void SetApplySoftmax( bool applySoftmax ) { Layer<CCrossEntropyLossLayer>()->SetApplySoftmax(applySoftmax); }
	bool GetApplySoftmax() const { return Layer<CCrossEntropyLossLayer>()->IsSoftmaxApplied(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "CrossEntropyLoss" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyBinaryCrossEntropyLossLayer : public CPyLossLayer {
public:
	explicit CPyBinaryCrossEntropyLossLayer( CBinaryCrossEntropyLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	void SetPositiveWeight( float value ) { Layer<CBinaryCrossEntropyLossLayer>()->SetPositiveWeight(value); }
	float GetPositiveWeight() const { return Layer<CBinaryCrossEntropyLossLayer>()->GetPositiveWeight(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "BinaryCrossEntropyLoss" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyEuclideanLossLayer : public CPyLossLayer {
public:
	explicit CPyEuclideanLossLayer( CEuclideanLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "EuclideanLoss" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyHingeLossLayer : public CPyLossLayer {
public:
	explicit CPyHingeLossLayer( CHingeLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "HingeLoss" );
		return pyConstructor( py::cast(this) );
	}
};


//------------------------------------------------------------------------------------------------------------

class CPySquaredHingeLossLayer : public CPyLossLayer {
public:
	explicit CPySquaredHingeLossLayer( CSquaredHingeLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "SquaredHingeLoss" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyMultiHingeLossLayer : public CPyLossLayer {
public:
	explicit CPyMultiHingeLossLayer( CMultiHingeLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MultiHingeLoss" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyMultiSquaredHingeLossLayer : public CPyLossLayer {
public:
	explicit CPyMultiSquaredHingeLossLayer( CMultiSquaredHingeLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "MultiSquaredHingeLoss" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyFocalLossLayer : public CPyLossLayer {
public:
	explicit CPyFocalLossLayer( CFocalLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	void SetForce( float value ) { Layer<CFocalLossLayer>()->SetFocalForce(value); }
	float GetForce() const { return Layer<CFocalLossLayer>()->GetFocalForce(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "FocalLoss" );
		return pyConstructor( py::cast(this), 0 );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyBinaryFocalLossLayer : public CPyLossLayer {
public:
	explicit CPyBinaryFocalLossLayer( CBinaryFocalLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	void SetForce( float value ) { Layer<CBinaryFocalLossLayer>()->SetFocalForce(value); }
	float GetForce() const { return Layer<CBinaryFocalLossLayer>()->GetFocalForce(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "BinaryFocalLoss" );
		return pyConstructor( py::cast(this), 0 );
	}
};

//------------------------------------------------------------------------------------------------------------

class CPyCenterLossLayer : public CPyLossLayer {
public:
	explicit CPyCenterLossLayer( CCenterLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	void SetClassCount( int value ) { Layer<CCenterLossLayer>()->SetNumberOfClasses(value); }
	int GetClassCount() const { return Layer<CCenterLossLayer>()->GetNumberOfClasses(); }
	void SetRate( float value ) { Layer<CCenterLossLayer>()->SetClassCentersConvergenceRate(value); }
	float GetRate() const { return Layer<CCenterLossLayer>()->GetClassCentersConvergenceRate(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "CenterLoss" );
		return pyConstructor( py::cast(this), 0, 0 );
	}
};

void InitializeLossLayer( py::module& m )
{
	py::class_<CPyLossLayer, CPyLayer>(m, "Loss")
		.def( "get_last_loss", &CPyLossLayer::GetLastLoss, py::return_value_policy::reference )

		.def( "get_loss_weight", &CPyLossLayer::GetLossWeight, py::return_value_policy::reference )
		.def( "set_loss_weight", &CPyLossLayer::SetLossWeight, py::return_value_policy::reference )

		.def( "get_train_labels", &CPyLossLayer::GetTrainLabels, py::return_value_policy::reference )
		.def( "set_train_labels", &CPyLossLayer::SetTrainLabels, py::return_value_policy::reference )

		.def( "get_max_gradient", &CPyLossLayer::GetMaxGradientValue, py::return_value_policy::reference )
		.def( "set_max_gradient", &CPyLossLayer::SetMaxGradientValue, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyCrossEntropyLossLayer, CPyLossLayer>(m, "CrossEntropyLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyCrossEntropyLossLayer( *layer.Layer<CCrossEntropyLossLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs,
			bool softmax, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CCrossEntropyLossLayer> loss = new CCrossEntropyLossLayer( mathEngine );
			loss->SetApplySoftmax( softmax );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "CrossEntropyLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyCrossEntropyLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_apply_softmax", &CPyCrossEntropyLossLayer::GetApplySoftmax, py::return_value_policy::reference )
		.def( "set_apply_softmax", &CPyCrossEntropyLossLayer::SetApplySoftmax, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyBinaryCrossEntropyLossLayer, CPyLossLayer>(m, "BinaryCrossEntropyLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyBinaryCrossEntropyLossLayer( *layer.Layer<CBinaryCrossEntropyLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float positiveWeight, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CBinaryCrossEntropyLossLayer> loss = new CBinaryCrossEntropyLossLayer( mathEngine );
			loss->SetPositiveWeight( positiveWeight );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "BinaryCrossEntropyLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyBinaryCrossEntropyLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_positive_weight", &CPyBinaryCrossEntropyLossLayer::GetPositiveWeight, py::return_value_policy::reference )
		.def( "set_positive_weight", &CPyBinaryCrossEntropyLossLayer::SetPositiveWeight, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyEuclideanLossLayer, CPyLossLayer>(m, "EuclideanLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEuclideanLossLayer( *layer.Layer<CEuclideanLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer( mathEngine );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "EuclideanLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyEuclideanLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyHingeLossLayer, CPyLossLayer>(m, "HingeLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyHingeLossLayer( *layer.Layer<CHingeLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CHingeLossLayer> loss = new CHingeLossLayer( mathEngine );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "EuclideanLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyHingeLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPySquaredHingeLossLayer, CPyLossLayer>(m, "SquaredHingeLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPySquaredHingeLossLayer( *layer.Layer<CSquaredHingeLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CSquaredHingeLossLayer> loss = new CSquaredHingeLossLayer( mathEngine );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "SquaredHingeLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPySquaredHingeLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMultiHingeLossLayer, CPyLossLayer>(m, "MultiHingeLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyMultiHingeLossLayer( *layer.Layer<CMultiHingeLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMultiHingeLossLayer> loss = new CMultiHingeLossLayer( mathEngine );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "MultiHingeLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyMultiHingeLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyMultiSquaredHingeLossLayer, CPyLossLayer>(m, "MultiSquaredHingeLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyMultiSquaredHingeLossLayer( *layer.Layer<CMultiSquaredHingeLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CMultiSquaredHingeLossLayer> loss = new CMultiSquaredHingeLossLayer( mathEngine );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "MultiSquaredHingeLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyMultiSquaredHingeLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyFocalLossLayer, CPyLossLayer>(m, "FocalLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyFocalLossLayer( *layer.Layer<CFocalLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float force, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CFocalLossLayer> loss = new CFocalLossLayer( mathEngine );
			loss->SetLossWeight( lossWeight );
			loss->SetFocalForce( force );
			loss->SetName( FindFreeLayerName( dnn, "FocalLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyFocalLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_force", &CPyFocalLossLayer::GetForce, py::return_value_policy::reference )
		.def( "set_force", &CPyFocalLossLayer::SetForce, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyBinaryFocalLossLayer, CPyLossLayer>(m, "BinaryFocalLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyBinaryFocalLossLayer( *layer.Layer<CBinaryFocalLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, float force, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CBinaryFocalLossLayer> loss = new CBinaryFocalLossLayer( mathEngine );
			loss->SetLossWeight( lossWeight );
			loss->SetFocalForce( force );
			loss->SetName( FindFreeLayerName( dnn, "BinaryFocalLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyBinaryFocalLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_force", &CPyBinaryFocalLossLayer::GetForce, py::return_value_policy::reference )
		.def( "set_force", &CPyBinaryFocalLossLayer::SetForce, py::return_value_policy::reference )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyCenterLossLayer, CPyLossLayer>(m, "CenterLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyCenterLossLayer( *layer.Layer<CCenterLossLayer>(), layer.MathEngineOwner() );
		}) )
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs, int classCount, float rate, float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CCenterLossLayer> loss = new CCenterLossLayer( mathEngine );
			loss->SetNumberOfClasses( classCount );
			loss->SetClassCentersConvergenceRate( rate );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "CenterLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyCenterLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
		.def( "get_rate", &CPyCenterLossLayer::GetRate, py::return_value_policy::reference )
		.def( "set_rate", &CPyCenterLossLayer::SetRate, py::return_value_policy::reference )
		.def( "get_class_count", &CPyCenterLossLayer::GetClassCount, py::return_value_policy::reference )
		.def( "set_class_count", &CPyCenterLossLayer::SetClassCount, py::return_value_policy::reference )
	;
}
