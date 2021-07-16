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
#include "PyMathEngine.h"

static const int PythonLossLayerVersion = 1;

static CArchive& operator <<( CArchive& archive, const py::object& obj )
{
	string str;
	{
		py::gil_scoped_acquire ac;
		py::object pyModule = py::module::import( "pickle" );
		py::object pyDumps = pyModule.attr( "dumps" );
		py::bytes dumpedObject = pyDumps( obj ).cast<py::bytes>();
		str = dumpedObject;
	}
	archive << static_cast<int>(str.length());
	for( size_t i = 0; i < str.length(); i++ ) {
		archive << str[i];
	}
	
	return archive;
}

static CArchive& operator >>( CArchive& archive, py::object& obj )
{
	int len = 0;
	archive >> len;
	string s(len, 't');
	for( size_t i = 0; i < len; i++ ) {
		archive >> s[i];
	}

	py::gil_scoped_acquire ac;
	py::bytes dumpedObject( s );
	py::object pyModule = py::module::import( "pickle" );
	py::object pyLoads = pyModule.attr( "loads" );
	obj = pyLoads(dumpedObject);

	return archive;
}

//------------------------------------------------------------------------------------------------------------

class CTempBlob : public CDnnBlob {
public:
	CTempBlob( IMathEngine& mathEngine, const CConstFloatHandle& data, const CBlobDesc& dataDesc );
};

CTempBlob::CTempBlob( IMathEngine& mathEngine, const CConstFloatHandle& data, const CBlobDesc& dataDesc ) :
	CDnnBlob( mathEngine, dataDesc, data, false )
{
}

//------------------------------------------------------------------------------------------------------------

class CPythonLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CPythonLossLayer )
public:
	CPythonLossLayer( IMathEngine& mathEngine ) :
		CLossLayer( mathEngine, "CCustomLossLayer" ) {}

	CPythonLossLayer( IMathEngine& mathEngine, const py::object& _lossCalculator ) :
		CLossLayer( mathEngine, "CCustomLossLayer" ), lossCalculator( _lossCalculator ) {}

protected:
	void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override
	{
		CGradientTape tape;

		CPtr<CPyMathEngineOwner> mathEngineOwner = new CPyMathEngineOwner( &MathEngine(), false );

		CPtr<const CDnnBlob> dataBlob = new CTempBlob( mathEngineOwner->MathEngine(), data, inputBlobs[0]->GetDesc() );
		CPtr<const CDnnBlob> var = tape.Variable( *dataBlob.Ptr() );
		CPyBlob dataPyBlob( *mathEngineOwner, const_cast<CDnnBlob*>(var.Ptr()) );

		CPtr<CDnnBlob> labelBlob( new CTempBlob( mathEngineOwner->MathEngine(), label, inputBlobs[1]->GetDesc() ) );
		CPyBlob labelPyBlob( *mathEngineOwner, labelBlob );

		CPtr<CDnnBlob> value;
		{
			py::gil_scoped_acquire acquire;
			py::object pyModule = py::module::import( "neoml.Dnn" );
			py::object pyFunction = pyModule.attr( "call_loss_calculator" );
			CPyBlob result = pyFunction( dataPyBlob, labelPyBlob, lossCalculator ).cast<CPyBlob>();
			value = result.Blob();
		}
		mathEngineOwner->MathEngine().VectorCopy( lossValue, value->GetData(), batchSize );
		if( !lossGradient.IsNull() ) {
			CPtr<const CDnnBlob> gradient = tape.Gradient( *value, *var );
			mathEngineOwner->MathEngine().VectorCopy( lossGradient, gradient->GetData(), batchSize * vectorSize );
		}
	}

	void BatchCalculateLossAndGradient(int, CConstFloatHandle, int, CConstFloatHandle, int, CFloatHandle, CFloatHandle, CFloatHandle) override
	{
		CheckArchitecture( false, GetName(), "The custom loss layer doesn't support label gradient calculation!" );
	}

	void BatchCalculateLossAndGradient(int, CConstFloatHandle, int, CConstIntHandle, int, CFloatHandle, CFloatHandle) override
	{
		CheckArchitecture( false, GetName(), "The custom loss layer doesn't support int labels!" );
	}

	void Serialize( CArchive& archive )
	{
		archive.SerializeVersion( PythonLossLayerVersion, 1 );
		CLossLayer::Serialize( archive );

		if( archive.IsLoading() ) {
			archive >> lossCalculator;
		} else {
			archive << lossCalculator;
		}
	}

private:
	py::object lossCalculator;
};

REGISTER_NEOML_LAYER( CPythonLossLayer, "NeoMLCustomLossLayer" )

//------------------------------------------------------------------------------------------------------------

class CPyCustomLossLayer : public CPyLossLayer {
public:
	explicit CPyCustomLossLayer( CPythonLossLayer& layer, CPyMathEngineOwner& mathEngineOwner ) :
		CPyLossLayer( layer, mathEngineOwner ) {}

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "CustomLoss" );
		return pyConstructor( py::cast(this) );
	}
};

//------------------------------------------------------------------------------------------------------------

void InitializeCustomLossLayer( py::module& m )
{
	py::class_<CPyCustomLossLayer, CPyLossLayer>(m, "CustomLoss")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyCustomLossLayer( *layer.Layer<CPythonLossLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const py::object& object, const std::string& name, const py::list& layers, const py::list& outputs,
			float lossWeight )
		{
			py::gil_scoped_release release;
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CPythonLossLayer> loss = new CPythonLossLayer( mathEngine, object );
			loss->SetLossWeight( lossWeight );
			loss->SetName( FindFreeLayerName( dnn, "CustomLoss", name ).c_str() );
			dnn.AddLayer( *loss );
			loss->Connect( 0, layers[0].cast<CPyLayer>().BaseLayer(), outputs[0].cast<int>() );
			loss->Connect( 1, layers[1].cast<CPyLayer>().BaseLayer(), outputs[1].cast<int>() );
			if( layers.size() == 3 ) {
				loss->Connect( 2, layers[2].cast<CPyLayer>().BaseLayer(), outputs[2].cast<int>() );
			}

			return CPyCustomLossLayer( *loss, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;
}
