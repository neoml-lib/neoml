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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/PCA.h>

namespace NeoML {

CPtr<IModel> CPCA::Train( const IProblem& problem )
{
#ifdef NEOML_USE_MKL
	float[12] a = { 1, 2, 3, 3, 2, 1, 2, 3, 4, 4, 3, 2 };
	float[9] vt;
	float* u = nullptr;
	float[16] superb;
	LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'O', 'A', 4, 3, &a, 4, u, 4, &vt, 3, &superb);
	printf("kulebaka\n");
#endif
	return nullptr;
}

} // namespace NeoML