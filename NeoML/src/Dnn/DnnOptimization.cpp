/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/DnnOptimization.h>
#include <NeoML/Dnn/Optimization/Graph.h>
#include "Optimization/BatchNormFusionOptimizer.h"
#include "Optimization/ChannelwiseWith1x1Optimizer.h"
#include "Optimization/MobileNetV2Optimizer.h"
#include "Optimization/MobileNetV3Optimizer.h"
#include "Optimization/OptimizerFunctions.h"
#include <NeoML/Dnn/Layers/RowwiseOperationChainLayer.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

CDnnOptimizationReport OptimizeDnn( CDnn& dnn, const CDnnOptimizationSettings& settings )
{
	CDnnOptimizationReport report;
	optimization::CGraph graph( dnn );

	report.UnpackedCompositeLayers = optimization::UnpackComposites( graph );
	report.RemovedTrivialLayers = optimization::RemoveTrivialLayers( graph );
	optimization::CBatchNormFusionOptimizer( graph ).Apply( report );

	if( settings.AllowCpuOnlyOptimizations ) {
		optimization::CChannelwiseWith1x1Optimizer( graph ).Apply( report );
		optimization::CMobileNetV2Optimizer( graph ).Apply( report );
		optimization::CMobileNetV3Optimizer( graph ).Apply( report );

		CArray<int> chains;
		OptimizeRowwiseChains(dnn, chains);
		report.RowwiseChainCount = chains.Size();
	}
	return report;
}

CDnnOptimizationReport OptimizeDnnOnLoad(CDnn& dnn, size_t size)
{
	CDnnOptimizationReport report;
	optimization::CGraph graph(dnn);

	report.UnpackedCompositeLayers = optimization::UnpackComposites(graph);
	report.RemovedTrivialLayers = optimization::RemoveTrivialLayers(graph);
	optimization::CBatchNormFusionOptimizer(graph).Apply(report);

	if (size < 1024 * 1024) {
		optimization::CMobileNetV2Optimizer(graph).Apply(report);
		
		CArray<int> chains;
		OptimizeRowwiseChains(dnn, chains);
		report.RowwiseChainCount = chains.Size();
	}

	return report;
}

} // namespace NeoML
