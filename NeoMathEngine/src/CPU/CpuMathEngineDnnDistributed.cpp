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

#include <CpuMathEngine.h>
#include <CpuMathEngineDnnDistributed.h>
#include <MemoryHandleInternal.h>

namespace NeoML {

CMultiThreadDistributedCommunicator::CMultiThreadDistributedCommunicator( int _n_threads )
    : counter( _n_threads ), waiting_flag( true ), n_threads( _n_threads )
{
    handles.resize( n_threads );
}

void CMultiThreadDistributedCommunicator::barrier()
{
    int wait = waiting_flag;
    if( !--counter ){
        counter = n_threads;
        waiting_flag++;
    } else {
        while( waiting_flag == wait ){
            std::this_thread::yield();
        }
    }
}

/*
void CMultiThreadDistributedCommunicator::barrier()
{
    std::unique_lock<std::mutex> lock(m);
    int wait = waiting_flag;
    if( !--counter ){
        waiting_flag++;
        counter = n_threads;
        cv.notify_all();
    } else {
        cv.wait(lock, [this, wait]{ return ( wait != waiting_flag ); } );
    }
}
*/

void CMultiThreadDistributedCommunicator::AllReduce( const CFloatHandle& handle, int size )
{
    IMathEngine* mathEngine = handle.GetMathEngine();
    int thread = mathEngine->GetDistributedInfo().Thread;
    handles[thread] = reinterpret_cast<float*>( GetRaw( handle ) );

    barrier();

    int perThread = ( size + n_threads - 1 ) / n_threads;
    float buf;
    for( int i = thread * perThread; i < max( ( thread + 1 ) * perThread, size ); i++ ){
        buf = 0;
        for( int j = 0; j < n_threads; j++ ){
            buf += handles[j][i];
        }
        buf /= n_threads;
        for( int j = 0; j < n_threads; j++ ){
            handles[j][i] = buf;
        }
    }
    
    barrier();
}

void CreateDistributedCpuMathEngines( IMathEngine** mathEngines, int count )
{
    auto comm = std::make_shared<CMultiThreadDistributedCommunicator>( count );
    for( int i = 0; i < count; i++ ){
        mathEngines[i] = CreateCpuMathEngine( 1, 0 );
        static_cast<CCpuMathEngine*>( mathEngines[i] )->SetDistributedCommunicator( comm, {i, count} );
    }
}

}