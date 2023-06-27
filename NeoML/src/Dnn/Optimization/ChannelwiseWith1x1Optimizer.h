/* Copyright © 2017-2023 ABBYY Production LLC

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

#pragma once

namespace NeoML {

// Forward declaration(s)
class CBaseLayer;
class CConvLayer;
class CChannelwiseConvLayer;
struct CDnnOptimizationReport;

namespace optimization {

// Forward declaration(s)
class CGraph;

class CChannelwiseWith1x1Optimizer {
public:
	explicit CChannelwiseWith1x1Optimizer( CGraph& graph ) :
		graph( graph )
	{
	}

	// Optimizes the graph and writes the result to the report
	void Apply( CDnnOptimizationReport& report );

private:
	CGraph& graph;

	int optimizeNonResidualBlocks();
	int optimizeResidualConnections();

	bool isValid1x1Conv( CConvLayer& conv ) const;
	bool isValidActivation( CBaseLayer& layer ) const;
	bool isValidChannelwise( CChannelwiseConvLayer& channelwise ) const;
};

} // namespace optimization

} // namespace NeoML
