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

#include "PyPCA.h"

static inline py::array getArray( const CArray<float>& matr )
{
	py::array_t<double, py::array::c_style> array( { matr.Size() } );
	auto tempArray = array.mutable_unchecked<1>();
	for( int i = 0; i < matr.Size(); i++ ){
		tempArray(i) = matr[i];
	}
	return array;
}

static CFloatMatrixDesc getMatrix( int height, int width, const int* columns, const float* values, const int* rowPtr )
{
	CSparseFloatMatrixDesc desc;
	desc.Height = height;
	desc.Width = width;
	desc.Columns = const_cast<int*>( columns );
	desc.Values = const_cast<float*>( values );
	desc.PointerB = const_cast<int*>( rowPtr );
	desc.PointerE = const_cast<int*>( rowPtr ) + 1;
	return desc;
}

class CPyPca : public CPca {
public:
	explicit CPyPca( const CPca::CParams& p ) : CPca( p ) {}

	void Fit( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse );
	py::array FitTransform( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse );
	py::array SingularValues() { return getArray( GetSingularValues() ); }
	py::array ExplainedVariance() { return getArray( GetExplainedVariance() ); }
	py::array ExplainedVarianceRatio() { return getArray( GetExplainedVarianceRatio() ); }
	py::array Components();
	int NComponents() { return GetComponentsNum(); }
	float NoiseVariance() { return GetNoiseVariance(); }
};

void CPyPca::Fit( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse )
{
	CFloatMatrixDesc desc = getMatrix( height, width,
		reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr ), reinterpret_cast<const float*>( data.data() ),
		reinterpret_cast<const int*>( rowPtr.data() ) );
	py::gil_scoped_release release;
	Train( desc );
}

py::array CPyPca::FitTransform( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse )
{
	CFloatMatrixDesc desc = getMatrix( height, width,
		reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr ), reinterpret_cast<const float*>( data.data() ),
		reinterpret_cast<const int*>( rowPtr.data() ) );

	CFloatMatrixDesc resDesc;
	{
		py::gil_scoped_release release;
		resDesc = Transform( desc );
	}
	py::array_t<double, py::array::c_style> transformed( { resDesc.Height, resDesc.Width } );
	memset( static_cast<double*>( transformed.request().ptr ), 0, resDesc.Height * resDesc.Width * sizeof(double) );
	auto tempTransformed = transformed.mutable_unchecked<2>();
	for( int i = 0; i < resDesc.Height; i++ ) {
		for( int j = resDesc.PointerB[i]; j < resDesc.PointerE[i]; j++ ){
			tempTransformed(i, resDesc.Columns[j]) = resDesc.Values[j];
		}
	}
	return transformed;
}

py::array CPyPca::Components()
{
	CFloatMatrixDesc desc = GetComponents();
	py::array_t<double, py::array::c_style> components( { desc.Height, desc.Width } );
	memset( static_cast<double*>( components.request().ptr ), 0, desc.Height * desc.Width * sizeof(double) );
	auto tempComponents = components.mutable_unchecked<2>();
	for( int i = 0; i < desc.Height; i++ ){
		for( int j = desc.PointerB[i]; j < desc.PointerE[i]; j++ ){
			tempComponents(i, desc.Columns[j]) = desc.Values[j];
		}
	}
	return components;
}

void InitializePCA(py::module& m)
{
	py::class_<CPyPca>(m, "PCA")
		.def( py::init(
			[]( const std::string& components_type, float n_components ) {
				CPca::CParams p;
				if( components_type == "None" ) {
					p.ComponentsType = CPca::TComponents::PCAC_None;
				} else if( components_type == "Int" ) {
					p.ComponentsType = CPca::TComponents::PCAC_Int;
				} else if( components_type == "Float" ) {
					p.ComponentsType = CPca::TComponents::PCAC_Float;
				}
				p.Components = n_components;
				return new CPyPca( p );
			})
		)

		.def( "fit", &CPyPca::Fit, py::return_value_policy::reference )
		.def( "fit_transform", &CPyPca::FitTransform, py::return_value_policy::reference )
		.def( "components", &CPyPca::Components, py::return_value_policy::reference )
		.def( "n_components", &CPyPca::NComponents, py::return_value_policy::reference )
		.def( "explained_variance", &CPyPca::ExplainedVariance, py::return_value_policy::reference )
		.def( "explained_variance_ratio", &CPyPca::ExplainedVarianceRatio, py::return_value_policy::reference )
		.def( "singular_values", &CPyPca::SingularValues, py::return_value_policy::reference )
		.def( "noise_variance", &CPyPca::NoiseVariance, py::return_value_policy::reference )
	;
}