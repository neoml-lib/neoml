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

#include "PyAutoDiff.h"

void InitializeTape(py::module& m)
{
	m.def( "tape_sum", [](const CPyBlob& first) {
		CPtr<const CDnnBlob> result( Sum( first.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("tape_add", [](const CPyBlob& first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Add( first.Blob(), second.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("tape_mult", [](const CPyBlob& first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Mult( first.Blob(), second.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );
}
