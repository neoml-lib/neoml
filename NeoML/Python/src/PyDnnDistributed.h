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

#pragma once

#include "PyDnn.h"
#include "PyDnnBlob.h"
#include <NeoML/Dnn/DnnDistributed.h>

class CPyDistributedDataset : public IDistributedDataset {
public:
    CPyDistributedDataset( const py::object& data ) : getData( data ) {};
    void SetInputBatch( CDnn& dnn, int thread ) override;
private:
    py::object getData;
};

class CPyDistributedTraining : public CDistributedTraining {
public:
    CPyDistributedTraining( CDnn& dnn, int count, TDistributedInitializer initializer, int seed )
        : CDistributedTraining( dnn, count, initializer, seed ) {};
    CPyDistributedTraining( CArchive& archive, int count, TDistributedInitializer initializer, int seed )
        : CDistributedTraining( archive, count, initializer, seed ) {};
    CPyDistributedTraining( CDnn& dnn, const CArray<int>& cudaDevs, TDistributedInitializer initializer, int seed )
        : CDistributedTraining( dnn, cudaDevs, initializer, seed ) {};
    CPyDistributedTraining( CArchive& archive, const CArray<int>& cudaDevs, TDistributedInitializer initializer, int seed )
        : CDistributedTraining( archive, cudaDevs, initializer, seed ) {};
    void Run( const py::object& data );
    void RunAndBackward( const py::object& data );
    void Learn( const py::object& data );
    void Train_();
    py::array LastLosses( const std::string& layer );
    py::list GetOutput( const std::string& layer );
    int GetModelCount_() { return GetModelCount(); }
    void SetSolver_( const std::string& path );
    void Save( const std::string& path );
};

void InitializeDistributedTraining(py::module& m);