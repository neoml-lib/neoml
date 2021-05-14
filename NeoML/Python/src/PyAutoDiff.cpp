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

std::vector<int> createShape( const py::array& data )
{
	std::vector<int> shape;
	switch( data.ndim() ) {
		case 1:
			shape = {1, 1, 1, 1, 1, 1, (int)data.shape(0)};
			break;
		case 2:
			shape = {1, 1, 1, 1, 1, (int)data.shape(0), (int)data.shape(1)};
			break;
		case 3:
			shape = {1, 1, 1, 1, (int)data.shape(0), (int)data.shape(1), (int)data.shape(2)};
			break;
		case 4:
			shape = {1, 1, 1, (int)data.shape(0), (int)data.shape(1), (int)data.shape(2), (int)data.shape(3)};
			break;
		case 5:
			shape = {1, 1, (int)data.shape(0), (int)data.shape(1), (int)data.shape(2), (int)data.shape(3), (int)data.shape(4)};
			break;
		case 6:
			shape = {1, (int)data.shape(0), (int)data.shape(1), (int)data.shape(2), (int)data.shape(3), (int)data.shape(4), (int)data.shape(5)};
			break;
		case 7:
			shape = {(int)data.shape(0), (int)data.shape(1), (int)data.shape(2), (int)data.shape(3), (int)data.shape(4), (int)data.shape(5), (int)data.shape(6)};
			break;
		default:
			assert( false );
	};
	return shape;
}

void InitializeTape(py::module& m)
{
	m.def("blob_const", []( const CPyMathEngine& mathEngine, py::array desc, float data ) {
		std::vector<int> shape = createShape( desc );
		CPtr<const CDnnBlob> result( Const( mathEngine.MathEngineOwner().MathEngine(), data,
			{shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]} ) );
		return CPyBlob( mathEngine.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	});

	m.def("blob_const", []( const CPyMathEngine& mathEngine, py::array desc, py::buffer buffer ) {
		py::buffer_info info = buffer.request();
		std::vector<int> shape = createShape( desc );
		CPtr<const CDnnBlob> result( Const( mathEngine.MathEngineOwner().MathEngine(), (float*)info.ptr,
			{shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]} ) );
		return CPyBlob( mathEngine.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	});

	m.def("blob_add", [](const CPyBlob& first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Add( first.Blob(), second.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_add", [](const CPyBlob& first, float second) {
		CPtr<const CDnnBlob> result( Add( first.Blob(), second ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_add", [](float first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Add( first, second.Blob() ) );
		return CPyBlob( second.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_sub", [](const CPyBlob& first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Sub( first.Blob(), second.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_sub", [](const CPyBlob& first, float second) {
		CPtr<const CDnnBlob> result( Sub( first.Blob(), second ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_sub", [](float first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Sub( first, second.Blob() ) );
		return CPyBlob( second.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_mult", [](const CPyBlob& first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Mult( first.Blob(), second.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_mult", [](const CPyBlob& first, float second) {
		CPtr<const CDnnBlob> result( Mult( first.Blob(), second ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_mult", [](float first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Mult( first, second.Blob() ) );
		return CPyBlob( second.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_div", [](const CPyBlob& first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Div( first.Blob(), second.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_div", [](const CPyBlob& first, float second) {
		CPtr<const CDnnBlob> result( Div( first.Blob(), second ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_div", [](float first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Div( first, second.Blob() ) );
		return CPyBlob( second.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_max", [](const CPyBlob& first, float second) {
		CPtr<const CDnnBlob> result( Max( first.Blob(), second ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_max", [](float first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Max( first, second.Blob() ) );
		return CPyBlob( second.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def( "blob_sum", [](const CPyBlob& first) {
		CPtr<const CDnnBlob> result( Sum( first.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def( "blob_neg", [](const CPyBlob& first) {
		CPtr<const CDnnBlob> result( Neg( first.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def( "blob_abs", [](const CPyBlob& first) {
		CPtr<const CDnnBlob> result( Abs( first.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def( "blob_log", [](const CPyBlob& first) {
		CPtr<const CDnnBlob> result( Log( first.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def( "blob_exp", [](const CPyBlob& first) {
		CPtr<const CDnnBlob> result( Exp( first.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_top_k", [](const CPyBlob& first, int k ) {
		CPtr<const CDnnBlob> result( TopK( first.Blob(), k ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_clip", [](const CPyBlob& first, float minValue, float maxValue) {
		CPtr<const CDnnBlob> result( Clip( first.Blob(), minValue, maxValue ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_binary_cross_entropy", [](const CPyBlob& first, const CPyBlob& second, bool fromLogits ) {
		CPtr<const CDnnBlob> result( BinaryCrossEntropy( first.Blob(), second.Blob(), fromLogits ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );
}
