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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_VULKAN

#include <VulkanMathEngine.h>
#include <NeoMathEngine/NeoMathEngineException.h>

namespace NeoML {

CLstmDesc* CVulkanMathEngine::InitLstm( int, int, const CConstFloatHandle&, const CConstFloatHandle&,
	const CConstFloatHandle&, const CConstFloatHandle& )
{
	ASSERT_EXPR( false );
	return nullptr;
}

void CVulkanMathEngine::Lstm( CLstmDesc&, bool, int, int, const CConstFloatHandle&, const CConstFloatHandle&,
	const CConstFloatHandle&, const CFloatHandle&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_VULKAN
