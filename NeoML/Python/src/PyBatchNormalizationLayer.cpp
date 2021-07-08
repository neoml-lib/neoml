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

#include "PyBatchNormalizationLayer.h"
#include "PyDnnBlob.h"

class CPyBatchNormalizationLayer : public CPyLayer {
public:
	explicit CPyBatchNormalizationLayer( CBatchNormalizationLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}

	void SetChannelBased(bool channelBased) { Layer<CBatchNormalizationLayer>()->SetChannelBased( channelBased ); }
	bool GetChannelBased() const { return Layer<CBatchNormalizationLayer>()->IsChannelBased(); }

	void SetCovergenceRate(float covergenceRate) { Layer<CBatchNormalizationLayer>()->SetSlowConvergenceRate( covergenceRate ); }
	float GetCovergenceRate() const { return Layer<CBatchNormalizationLayer>()->GetSlowConvergenceRate(); }

	void SetZeroFreeTerm(bool freeTerm) { Layer<CBatchNormalizationLayer>()->SetZeroFreeTerm( freeTerm ); }
	bool GetZeroFreeTerm() const { return Layer<CBatchNormalizationLayer>()->IsZeroFreeTerm(); }

	void SetFinalParams( const CPyBlob& blob ) { Layer<CBatchNormalizationLayer>()->SetFinalParams( blob.Blob() ); }
	CPyBlob GetFinalParams() const { return CPyBlob( MathEngineOwner(), Layer<CBatchNormalizationLayer>()->GetFinalParams() ); }

	void SetUseFinalParams(bool useFinalParams) { Layer<CBatchNormalizationLayer>()->UseFinalParamsForInitialization( useFinalParams ); }
	bool GetUseFinalParams() const { return Layer<CBatchNormalizationLayer>()->IsUsingFinalParamsForInitialization(); }

	py::object CreatePythonObject() const
	{
		py::object pyModule = py::module::import( "neoml.Dnn" );
		py::object pyConstructor = pyModule.attr( "BatchNormalization" );
		return pyConstructor( py::cast(this), 0 );
	}
};

void InitializeBatchNormalizationLayer( py::module& m )
{
	py::class_<CPyBatchNormalizationLayer, CPyLayer>(m, "BatchNormalization")
		.def( py::init([]( const CPyLayer& layer )
		{
			return new CPyBatchNormalizationLayer( *layer.Layer<CBatchNormalizationLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const CPyLayer& layer, int outputNumber, int channelBased, bool zeroFreeTerm, float slowConvergenceRate ) {
			py::gil_scoped_release release;
			CDnn& dnn = layer.Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();
			CPtr<CBatchNormalizationLayer> batchNorm = new CBatchNormalizationLayer( mathEngine );
			batchNorm->SetChannelBased( channelBased );
			batchNorm->SetZeroFreeTerm( zeroFreeTerm );
			batchNorm->SetSlowConvergenceRate( slowConvergenceRate );
			batchNorm->SetName( FindFreeLayerName( dnn, "BatchNormalization", name ).c_str() );
			dnn.AddLayer( *batchNorm );
			batchNorm->Connect( 0, layer.BaseLayer(), outputNumber );
			return new CPyBatchNormalizationLayer( *batchNorm, layer.MathEngineOwner() );
		}) )
		.def( "get_channel_based", &CPyBatchNormalizationLayer::GetChannelBased, py::return_value_policy::reference )
		.def( "set_channel_based", &CPyBatchNormalizationLayer::SetChannelBased, py::return_value_policy::reference )
		.def( "get_slow_convergence_rate", &CPyBatchNormalizationLayer::GetCovergenceRate, py::return_value_policy::reference )
		.def( "set_slow_convergence_rate", &CPyBatchNormalizationLayer::SetCovergenceRate, py::return_value_policy::reference )
		.def( "get_zero_free_term", &CPyBatchNormalizationLayer::GetZeroFreeTerm, py::return_value_policy::reference )
		.def( "set_zero_free_term", &CPyBatchNormalizationLayer::SetZeroFreeTerm, py::return_value_policy::reference )
		.def( "get_use_final_params_for_init", &CPyBatchNormalizationLayer::GetUseFinalParams, py::return_value_policy::reference )
		.def( "set_use_final_params_for_init", &CPyBatchNormalizationLayer::SetUseFinalParams, py::return_value_policy::reference )
	;
}
