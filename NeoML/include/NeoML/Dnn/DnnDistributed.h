/* Copyright © 2017-2020 ABBYY Production LLC
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

#include <memory>
#include <vector>
#include <NeoML/NeoML.h>

namespace NeoML {

class IDistributedDataset {
public:
	virtual void SetInputBatch( CDnn& cnn, int iteration, int thread ) = 0;
};

class NEOML_API CDistributedTraining : public IObject {
public:
	explicit CDistributedTraining( CArchive& archive, TMathEngineType type, int count, CArray<int> devs = {} );

	void RunAndLearnOnce( IDistributedDataset& data );
	float GetLastLoss( const CString& layerName );
private:
	std::vector<std::unique_ptr<IMathEngine>> mathEngines;
	std::vector<std::unique_ptr<CRandom>> rands;
	std::vector<std::unique_ptr<CDnn>> cnns;
};

} // namespace NeoML