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
#include "PyMemoryFile.h"

static inline py::array getArray( const CArray<float>& matr )
{
	py::array_t<float, py::array::c_style> array( { matr.Size() } );
	memcpy( static_cast<float*>( array.request().ptr ), matr.GetPtr(), matr.Size() * sizeof( float ) );
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
	CPyPca( const CPca::CParams& p ) : CPca( p ) {}

	void Fit( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse );
	py::array FitTransform( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse );
	py::array Transform_( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse );
	py::array SingularValues() { return getArray( GetSingularValues() ); }
	py::array ExplainedVariance() { return getArray( GetExplainedVariance() ); }
	py::array ExplainedVarianceRatio() { return getArray( GetExplainedVarianceRatio() ); }
	py::array Components();
	int NComponents() { return GetComponentsNum(); }
	float NoiseVariance() { return GetNoiseVariance(); }
	void Store( const std::string& path );
	void Load( const std::string& path );
};

void CPyPca::Load( const std::string& path )
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::load );
	CArchive archive( &file, CArchive::load );
	Serialize( archive );
}

void CPyPca::Store( const std::string& path )
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::store );
	CArchive archive( &file, CArchive::store );
	Serialize( archive );
}

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
		resDesc = TrainTransform( desc );
	}
	py::array_t<float, py::array::c_style> transformed( { resDesc.Height, resDesc.Width } );
	memset( static_cast<float*>( transformed.request().ptr ), 0, resDesc.Height * resDesc.Width * sizeof( float ) );
	auto tempTransformed = transformed.mutable_unchecked<2>();
	for( int i = 0; i < resDesc.Height; i++ ) {
		for( int j = resDesc.PointerB[i]; j < resDesc.PointerE[i]; j++ ){
			tempTransformed(i, resDesc.Columns[j]) = resDesc.Values[j];
		}
	}
	return transformed;
}

py::array CPyPca::Transform_( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse )
{
	CFloatMatrixDesc desc = getMatrix( height, width,
		reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr ), reinterpret_cast<const float*>( data.data() ),
		reinterpret_cast<const int*>( rowPtr.data() ) );

	CFloatMatrixDesc resDesc;
	{
		py::gil_scoped_release release;
		resDesc = Transform( desc );
	}
	py::array_t<float, py::array::c_style> transformed( { resDesc.Height, resDesc.Width } );
	memset( static_cast<float*>( transformed.request().ptr ), 0, resDesc.Height * resDesc.Width * sizeof( float ) );
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
	CSparseFloatMatrix matrix = GetComponents();
	CFloatMatrixDesc desc = matrix.GetDesc();
	py::array_t<float, py::array::c_style> components( { desc.Height, desc.Width } );
	memset( static_cast<float*>( components.request().ptr ), 0, desc.Height * desc.Width * sizeof( float ) );
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
			[]( const std::string& components_type, float n_components, bool isFullAlgorithm ) {
				CPca::CParams p;
				if( components_type == "None" ) {
					p.ComponentsType = CPca::TComponents::PCAC_None;
				} else if( components_type == "Int" ) {
					p.ComponentsType = CPca::TComponents::PCAC_Int;
				} else if( components_type == "Float" ) {
					p.ComponentsType = CPca::TComponents::PCAC_Float;
				}
				p.Components = n_components;
				p.SvdSolver = ( isFullAlgorithm ) ? SVD_Full : SVD_Randomized;
				return new CPyPca( p );
			})
		)

		.def( "store", &CPyPca::Store, py::return_value_policy::reference )
		.def( "load", &CPyPca::Load, py::return_value_policy::reference )
		.def( "fit", &CPyPca::Fit )
		.def( "fit_transform", &CPyPca::FitTransform, py::return_value_policy::reference )
		.def( "transform", &CPyPca::Transform_, py::return_value_policy::reference )
		.def( "components", &CPyPca::Components, py::return_value_policy::reference )
		.def( "n_components", &CPyPca::NComponents, py::return_value_policy::reference )
		.def( "explained_variance", &CPyPca::ExplainedVariance, py::return_value_policy::reference )
		.def( "explained_variance_ratio", &CPyPca::ExplainedVarianceRatio, py::return_value_policy::reference )
		.def( "singular_values", &CPyPca::SingularValues, py::return_value_policy::reference )
		.def( "noise_variance", &CPyPca::NoiseVariance, py::return_value_policy::reference )
		.def( py::pickle(
			[]( CPyPca& pyModel ) {
				CPyMemoryFile file;
				CArchive archive( &file, CArchive::store );
				pyModel.Serialize( archive );
				archive.Close();
				file.Close();
				return py::make_tuple( file.GetBuffer() );
			},
			[]( py::tuple t ) {
				if( t.size() != 1 ) {
					throw std::runtime_error("Invalid state!");
				}

				auto t0_array = t[0].cast<py::array>();
				CPyMemoryFile file( t0_array );
				CArchive archive( &file, CArchive::load );
				CPca* pcaSerialized = new CPca();
				pcaSerialized->Serialize( archive );
				return static_cast<CPyPca*>( pcaSerialized );
			}
		))
	;

	m.def( "singular_value_decomposition", []( int height, int width, py::array indices, py::array data, py::array rowPtr, bool isSparse,
		bool returnLeftVectors, bool returnRightVectors, bool isFullAlgorithm, int components ) {
		CFloatMatrixDesc desc = getMatrix( height, width, 
			reinterpret_cast<const int*>( isSparse ? indices.data() : nullptr ), reinterpret_cast<const float*>( data.data() ),
			reinterpret_cast<const int*>( rowPtr.data() ) );

		CArray<float> leftVectors;
		CArray<float> singularValues;
		CArray<float> rightVectors;
		{
			py::gil_scoped_release release;
			if( isFullAlgorithm ) {
				SingularValueDecomposition( desc, leftVectors, singularValues, rightVectors,
					returnLeftVectors, returnRightVectors, components );
			} else {
				RandomizedSingularValueDecomposition( desc, leftVectors, singularValues, rightVectors,
					returnLeftVectors, returnRightVectors, components );
			}
		}
		py::array_t<float, py::array::c_style> leftArray;
		if( returnLeftVectors ) {
			leftArray.resize( { height, components } );
			memcpy( static_cast<float*>( leftArray.request().ptr ), leftVectors.GetPtr(), height * components * sizeof( float ) );
		}
		py::array_t<float, py::array::c_style> singularArray( { components } );
		memcpy( static_cast<float*>( singularArray.request().ptr ), singularValues.GetPtr(), components * sizeof( float ) );
		py::array_t<float, py::array::c_style> rightArray;
		if( returnRightVectors ) {
			rightArray.resize( { components, width } );
			memcpy( static_cast< float* >( rightArray.request().ptr ), rightVectors.GetPtr(), components * width * sizeof( float ) );
		}
		return py::make_tuple( leftArray, singularArray, rightArray );
	}, py::return_value_policy::reference );
}