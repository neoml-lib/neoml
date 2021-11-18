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

#include <map>
#include "PyDnnDistributed.h"
#include "PyDnnBlob.h"
#include <NeoML/Dnn/DnnDistributed.h>

class CPyDistributedDataset : public IDistributedDataset {
public:
    CPyDistributedDataset( py::list _inputs ) : inputs( _inputs ) {};
    void SetInputBatch( CDnn& dnn, int thread ) override;
private:
    py::list inputs;
};

void CPyDistributedDataset::SetInputBatch( CDnn& dnn, int thread )
{
    CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );
    auto dnnInputs = inputs[thread].cast<std::map<std::string, CPyBlob>>();

    for( std::map<std::string, CPyBlob>::iterator it = dnnInputs.begin(); it != dnnInputs.end(); it++ ){
		CPtr<CSourceLayer> layer = dynamic_cast<CSourceLayer*>( dnn.GetLayer( it->first.c_str() ).Ptr() );
        CPyBlob input = it->second;
		layer->SetBlob( input.Blob() );
	}
}

class CPyDistributedTraining : public CDistributedTraining {
public:
    CPyDistributedTraining( CArchive& archive, int count )
        : CDistributedTraining( archive, count ) {};
    CPyDistributedTraining( CArchive& archive, const CArray<int>& cudaDevs )
        : CDistributedTraining( archive, cudaDevs ) {};
    void Learn( py::list inputs );
};

void CPyDistributedTraining::Learn( py::list inputs )
{
    CPyDistributedDataset dataset( inputs );
    RunAndLearnOnce( dataset );
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

        .def( "learn", &CPyDistributedTraining::Learn, py::return_value_policy::reference )
	;
} 