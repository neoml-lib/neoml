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

#include "PyDnnDistributed.h"

void CPyDistributedDataset::SetInputBatch( CDnn& dnn, int thread )
{
    py::gil_scoped_acquire acquire;

    CPyMathEngineOwner* owner = new CPyMathEngineOwner( &dnn.GetMathEngine(), false );
    CPyMathEngine mathEngine( *owner );
    py::object pyMathEngine = py::module::import( "neoml.MathEngine" ).attr( "MathEngine" )( mathEngine );
    py::dict inputs = getData( pyMathEngine, thread );

    for ( std::pair<py::handle, py::handle> item : inputs ){
        auto layerName = item.first.cast<std::string>();
        auto input = item.second.attr( "_internal" ).cast<CPyBlob>();
        CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn.GetLayer( layerName.c_str() ).Ptr() );
        layer->SetBlob( input.Blob() );
    }
}

void CPyDistributedTraining::Learn( const py::object& data )
{
    py::gil_scoped_release release;
    CPyDistributedDataset dataset( data );
    RunAndLearnOnce( dataset );
}

py::array CPyDistributedTraining::LastLosses( const std::string& layer )
{
    CArray<float> losses;
    GetLastLoss( layer, losses );
    py::array_t<double, py::array::c_style> lastLosses( { losses.Size() } );
    auto tempLosses = lastLosses.mutable_unchecked<1>();
    for( int i = 0; i < losses.Size(); i++ ){
        tempLosses(i) = losses[i];
    }
    return lastLosses;
}

void CPyDistributedTraining::Save( const std::string& path )
{
    CArchiveFile file( path.c_str(), CArchive::store );
    CArchive archive( &file, CArchive::store );
    Serialize( archive );
}

void InitializeDistributedTraining(py::module& m)
{
	py::class_<CPyDistributedTraining>(m, "DnnDistributed")
        .def( py::init(
            []( const std::string& path, int count ) {
                CArchiveFile file( path.c_str(), CArchive::load );
                CArchive archive( &file, CArchive::load );
                return new CPyDistributedTraining( archive, count );
            })
        )

        .def( py::init(
            []( const std::string& path, py::list cudaDevs ) {
                CArchiveFile file( path.c_str(), CArchive::load );
                CArchive archive( &file, CArchive::load );
                CArray<int> devs;
                for( int i = 0; i < cudaDevs.size(); i++ ){
                    devs.Add( cudaDevs[i].cast<int>() );
                }
                return new CPyDistributedTraining( archive, devs );
            })
        )

        .def( "_learn", &CPyDistributedTraining::Learn )
        .def( "_last_losses", &CPyDistributedTraining::LastLosses, py::return_value_policy::reference )
        .def( "_save", &CPyDistributedTraining::Save )
	;
} 