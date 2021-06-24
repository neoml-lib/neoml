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

std::vector<int> createShape( const py::array& desc )
{
	if( desc.size() > 7 ) {
		assert(false);
	}
	const int* shape = reinterpret_cast<const int*>( desc.data() );
	std::vector<int> resultShape( 7, 1 );
	for( int i = 7 - desc.size(), j = 0; i < 7; i++, j++ ) {
		resultShape[i] = shape[j];
	}
	return resultShape;
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

	m.def("blob_mul", [](const CPyBlob& first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Mul( first.Blob(), second.Blob() ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_mul", [](const CPyBlob& first, float second) {
		CPtr<const CDnnBlob> result( Mul( first.Blob(), second ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def("blob_mul", [](float first, const CPyBlob& second) {
		CPtr<const CDnnBlob> result( Mul( first, second.Blob() ) );
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

	m.def( "blob_sum", [](const CPyBlob& first, py::array axis) {
		CArray<int> axes;
		const int* axisPtr = reinterpret_cast< const int* >( axis.data() );
		axes.SetSize( axis.size() );
		for( int i = 0; i < axes.Size(); i++ ) {
			axes[i] = axisPtr[i];
		}
		CPtr<const CDnnBlob> result( Sum( first.Blob(), axes ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def( "blob_cumsum", [](const CPyBlob& first, int axis) {
		CPtr<const CDnnBlob> result( CumSum( first.Blob(), axis ) );
		return CPyBlob( first.MathEngineOwner(), const_cast<CDnnBlob*>(result.Ptr()) );
	}, py::return_value_policy::reference );

	m.def( "blob_mean", [](const CPyBlob& first, py::array axis) {
		CArray<int> axes;
		const int* axisPtr = reinterpret_cast< const int* >( axis.data() );
		axes.SetSize( axis.size() );
		for( int i = 0; i < axes.Size(); i++ ) {
			axes[i] = axisPtr[i];
		}
		CPtr<const CDnnBlob> result( Mean( first.Blob(), axes ) );
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
