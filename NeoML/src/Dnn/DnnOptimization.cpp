/* Copyright © 2017-2022 ABBYY Production LLC

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
#include "Optimization/ChannelwiseWith1x1Optimizer.h"
#include "Optimization/MobileNetV2Optimizer.h"
#include "Optimization/MobileNetV3Optimizer.h"
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

CDnnOptimizationReport OptimizeDnn( CDnn& dnn )
{
	CDnnOptimizationReport report;
	optimization::CGraph graph( dnn );
	optimization::CChannelwiseWith1x1Optimizer( graph ).Apply( report );
	optimization::CMobileNetV2Optimizer( graph ).Apply( report );
	optimization::CMobileNetV3Optimizer( graph ).Apply( report );
	return report;
}

} // namespace NeoML
