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

#include <thread>
#include <functional>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/DnnDistributed.h>

namespace NeoML {

CDistributedTraining::CDistributedTraining( CArchive& archive, TMathEngineType type, int count, std::initializer_list<int> devs )
{
    CreateDistributedMathEngines( mathEngines, type, count, devs );
    for( int i = 0; i < count; i++ ){
        rands.emplace_back( new CRandom( 42 ) );
        cnns.emplace_back( new CDnn( *rands[i], *mathEngines[i] ) );
        archive.Serialize( *cnns[i] );
        archive.Seek( 0, static_cast<CBaseFile::TSeekPosition>( 0 ) );
    }
}

void CDistributedTraining::RunAndLearnOnce( IDistributedDataset& data )
{
    std::vector<std::thread> threads;
    for ( unsigned i = 0; i < cnns.size(); i++ ) {
        std::thread t( std::bind(
            [&]( int thread ){
                data.SetInputBatch( *cnns[thread], 0, thread );
                cnns[thread]->RunAndLearnOnce();
            },  i ) );
        threads.push_back( std::move( t ) );
    }
    for ( unsigned i = 0; i < cnns.size(); i++ ) {
        threads[i].join();
    }
}

float CDistributedTraining::GetLastLoss( const CString& layerName )
{
    float loss = 0;
    for( unsigned i = 0; i < cnns.size(); i++ ){
        CLossLayer* lossLayer = dynamic_cast<CLossLayer*>( cnns[i]->GetLayer( layerName ).Ptr() );
        if( lossLayer == nullptr ){
            loss += dynamic_cast<CCtcLossLayer*>( cnns[i]->GetLayer( layerName ).Ptr() )->GetLastLoss();
        } else {
            loss += lossLayer->GetLastLoss();
        }
    }
    return loss / cnns.size();
}

} // namespace NeoML