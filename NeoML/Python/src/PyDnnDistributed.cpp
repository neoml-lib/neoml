/* Copyright © 2017-2024 ABBYY

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

#include "PyDnnDistributed.h"

int CPyDistributedDataset::SetInputBatch( CDnn& dnn, int thread )
{
	py::gil_scoped_acquire acquire;

	CPyMathEngineOwner* owner = new CPyMathEngineOwner( &dnn.GetMathEngine(), false );
	CPyMathEngine mathEngine( *owner );
	py::object pyMathEngine = py::module::import( "neoml.MathEngine" ).attr( "MathEngine" )( mathEngine );
	py::tuple input_data = py::tuple( getData( pyMathEngine, thread ) );
	const int batchSize = py::int_( input_data[0] );
	py::dict inputs = py::dict( input_data[1] );

	for ( std::pair<py::handle, py::handle> item : inputs ){
		auto layerName = item.first.cast<std::string>();
		auto input = item.second.attr( "_internal" ).cast<CPyBlob>();
		CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn.GetLayer( layerName.c_str() ).Ptr() );
		layer->SetBlob( input.Blob() );
	}

	return batchSize;
}

void CPyDistributedTraining::Run( const py::object& data )
{
	py::gil_scoped_release release;
	CPyDistributedDataset dataset( data );
	RunOnce( dataset );
}

void CPyDistributedTraining::RunAndBackward( const py::object& data )
{
	py::gil_scoped_release release;
	CPyDistributedDataset dataset( data );
	RunAndBackwardOnce( dataset );
}

void CPyDistributedTraining::Learn( const py::object& data )
{
	py::gil_scoped_release release;
	CPyDistributedDataset dataset( data );
	RunAndLearnOnce( dataset );
}

void CPyDistributedTraining::Train_()
{
	py::gil_scoped_release release;
	Train();
}

py::array CPyDistributedTraining::LastLosses( const std::string& layer )
{
	CArray<float> losses;
	GetLastLoss( layer, losses );
	py::array_t<float, py::array::c_style> lastLosses( py::ssize_t{ losses.Size() } );
	NeoAssert( losses.Size() == lastLosses.size() );
	memcpy( static_cast<float*>( lastLosses.request().ptr ), losses.GetPtr(), losses.Size() * sizeof( float ) );
	return lastLosses;
}

py::list CPyDistributedTraining::GetOutput( const std::string& layer )
{
	CObjectArray<CDnnBlob> blobs;
	GetLastBlob( layer, blobs );
	py::list output( blobs.Size() );

	CPtr<CPyMathEngineOwner> owner = new CPyMathEngineOwner( &GetDefaultCpuMathEngine(), false );
	for( int i = 0; i < blobs.Size(); i++ ){
		CPtr<CDnnBlob> blob = CDnnBlob::CreateBlob( GetDefaultCpuMathEngine(), CT_Float, blobs[i]->GetDesc() );
		blob->CopyFrom( blobs[i] );
		output[i] = CPyBlob( *owner, blob );
	}
	return output;
}

void CPyDistributedTraining::SetSolver_( const std::string& path )
{
	CArchiveFile file( path.c_str(), CArchive::load );
	CArchive archive( &file, CArchive::load );
	SetSolver( archive );
}

void CPyDistributedTraining::Save( const std::string& path )
{
	py::gil_scoped_release release;
	CArchiveFile file( path.c_str(), CArchive::store );
	CArchive archive( &file, CArchive::store );
	Serialize( archive );
}

static TDistributedInitializer getInitializer( const std::string& initializerName )
{
	if( initializerName == "xavier" ) {
		return TDistributedInitializer::Xavier;
	}
	if( initializerName == "xavier_uniform" ) {
		return TDistributedInitializer::XavierUniform;
	}
	return TDistributedInitializer::Uniform;
}

void InitializeDistributedTraining(py::module& m)
{
	py::class_<CPyDistributedTraining>(m, "DnnDistributed")
		.def( py::init(
			[]( const std::string& path, int count, const std::string& initializerName, int seed ) {
				CArchiveFile file( path.c_str(), CArchive::load );
				CArchive archive( &file, CArchive::load );
				return new CPyDistributedTraining( archive, count, getInitializer( initializerName ), seed );
			})
		)

		.def( py::init(
			[]( CPyDnn& dnn, int count, const std::string& initializerName, int seed ) {
				return new CPyDistributedTraining( dnn.Dnn(), count, getInitializer( initializerName ), seed );
			})
		)

		.def( py::init(
			[]( const std::string& path, py::list cudaDevs, const std::string& initializerName, int seed ) {
				CArchiveFile file( path.c_str(), CArchive::load );
				CArchive archive( &file, CArchive::load );
				CArray<int> devs;
				for( int i = 0; i < cudaDevs.size(); i++ ){
					devs.Add( cudaDevs[i].cast<int>() );
				}
				return new CPyDistributedTraining( archive, devs, getInitializer( initializerName ), seed );
			})
		)

		.def( py::init(
			[]( CPyDnn& dnn, py::list cudaDevs, const std::string& initializerName, int seed ) {
				CArray<int> devs;
				for( int i = 0; i < cudaDevs.size(); i++ ){
					devs.Add( cudaDevs[i].cast<int>() );
				}
				return new CPyDistributedTraining( dnn.Dnn(), devs, getInitializer( initializerName ), seed );
			})
		)

		.def( "_run", &CPyDistributedTraining::Run )
		.def( "_run_and_backward", &CPyDistributedTraining::RunAndBackward )
		.def( "_learn", &CPyDistributedTraining::Learn )
		.def( "_train", &CPyDistributedTraining::Train_ )
		.def( "_last_losses", &CPyDistributedTraining::LastLosses, py::return_value_policy::reference )
		.def( "_get_output", &CPyDistributedTraining::GetOutput, py::return_value_policy::reference )
		.def( "_get_model_count", &CPyDistributedTraining::GetModelCount, py::return_value_policy::reference )
		.def( "_set_solver", &CPyDistributedTraining::SetSolver_, py::return_value_policy::reference )
		.def( "_save", &CPyDistributedTraining::Save )
	;
}
