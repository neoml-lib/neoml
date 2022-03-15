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

#include <NeoML/Dnn/Layers/LossLayer.h>
#include <NeoML/Dnn/Layers/SinkLayer.h>
#include <NeoML/Dnn/Layers/CtcLayer.h>
#include <NeoML/Dnn/Layers/CrfLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/DnnDistributed.h>

#if FINE_PLATFORM( FINE_WINDOWS )
#include <Windows.h>
#endif

namespace NeoML {

#if FINE_PLATFORM( FINE_WINDOWS )

typedef BOOL (WINAPI *TGetProcessorInfoFunc) (LOGICAL_PROCESSOR_RELATIONSHIP,
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, PDWORD);
typedef BOOL (WINAPI *TSetThreadGroupFunc) (HANDLE, const GROUP_AFFINITY*, PGROUP_AFFINITY);

static TGetProcessorInfoFunc getProcessorInfo = nullptr;
static TSetThreadGroupFunc setThreadGroupAffinity = nullptr;
static bool winApiFunctionsLoaded = false;
static CCriticalSection functionLoadSection;

void loadWinAPIFunctions()
{
    CCriticalSectionLock lock( functionLoadSection );
    if( winApiFunctionsLoaded ) {
        return;
    }

    winApiFunctionsLoaded = true;
    HMODULE kernel32Handle = ::GetModuleHandle( L"kernel32" );
    if( kernel32Handle == NULL ) {
        return;
    }

    getProcessorInfo = ( TGetProcessorInfoFunc ) ::GetProcAddress( kernel32Handle, "GetLogicalProcessorInformationEx" );
    if( getProcessorInfo == nullptr ) {
        return;
    }
    setThreadGroupAffinity = ( TSetThreadGroupFunc ) ::GetProcAddress( kernel32Handle, "SetThreadGroupAffinity" );
    if( setThreadGroupAffinity == nullptr ) {
        getProcessorInfo = nullptr;
    }
}

static int getPhysicalCpuCount()
{
    loadWinAPIFunctions();
    if( getProcessorInfo == nullptr ) {
        return 1;
    }

    DWORD bufferSize = 0;
    getProcessorInfo( RelationProcessorPackage, nullptr, &bufferSize );
    NeoAssert( bufferSize > 0 );

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = static_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>( ::malloc( bufferSize ) );
    NeoAssert( buffer != nullptr );

    DWORD result = getProcessorInfo( RelationProcessorPackage, buffer, &bufferSize );
    if( result == 0 ) {
        ::free( buffer );
        NeoAssert( false );
        return 1;
    }
    
    int processorPackageCount = 0;
    DWORD offset = 0;
    while( offset < bufferSize ) {
        processorPackageCount++;
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* currInfo = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            reinterpret_cast<char*>( buffer ) + offset );
        offset += currInfo->Size;
    }
    free( buffer );
    return processorPackageCount;
}

#else

static int getPhysicalCpuCount()
{
    return 1;
}

#endif

// RAII switcher of current thread's group
class CThreadGroupSwitcher {
public:
#if FINE_PLATFORM( FINE_WINDOWS )
    CThreadGroupSwitcher( int threadIndex, int threadCount, int physicalCpuCount ) :
        physicalCpuCount( physicalCpuCount )
    {
        if( physicalCpuCount > 1 ) {
            NeoAssert( setThreadGroupAffinity != nullptr );
            const int threadPerCpu = ( threadCount + physicalCpuCount - 1 ) / physicalCpuCount;
            GROUP_AFFINITY affinity;
            affinity.Reserved[0] = 0;
            affinity.Reserved[1] = 0;
            affinity.Reserved[2] = 0;
            affinity.Group = threadIndex / threadPerCpu;
            affinity.Mask = static_cast<KAFFINITY>( 1ULL << ( threadIndex % threadPerCpu ) );
            NeoAssert( setThreadGroupAffinity( ::GetCurrentThread(), &affinity, &prevAffinity ) != 0 );
        }
    }

    ~CThreadGroupSwitcher()
    {
        if( physicalCpuCount > 1 && setThreadGroupAffinity != nullptr ) {
            setThreadGroupAffinity( ::GetCurrentThread(), &prevAffinity, nullptr );
        }
    }
#else
    CThreadGroupSwitcher( int, int, int ) {}
    ~CThreadGroupSwitcher() = default;
#endif

    CThreadGroupSwitcher( const CThreadGroupSwitcher& ) = delete;
    CThreadGroupSwitcher operator=( const CThreadGroupSwitcher& ) = delete;

private:
#if FINE_PLATFORM( FINE_WINDOWS )
    const int physicalCpuCount;
    GROUP_AFFINITY prevAffinity;
#endif
};

static CPtr<CDnnInitializer> createInitializer( TDistributedInitializer type, CRandom& random )
{
    switch( type ) {
        case TDistributedInitializer::Xavier:
            return new CDnnXavierInitializer( random );
        case TDistributedInitializer::XavierUniform:
            return new CDnnXavierUniformInitializer( random );
        case TDistributedInitializer::Uniform:
            return new CDnnUniformInitializer( random );
        default:
            NeoAssert( false );
    }
    return nullptr;
}

void CDistributedTraining::initialize( CArchive& archive, int count, TDistributedInitializer initializer, int seed )
{
    NeoAssert( archive.IsLoading() );
    for( int i = 0; i < count; i++ ){
        rands.Add( new CRandom( seed ) );
        cnns.Add( new CDnn( *rands[i], *mathEngines[i] ) );
        cnns[i]->SetInitializer( createInitializer( initializer, *rands[i] ) );
        cnns[i]->SetInitializer( new CDnnDistributedInitializer( *rands[i], mathEngines[i], cnns[i]->GetInitializer() ) );
        archive.Serialize( *cnns[i] );
        archive.Seek( 0, static_cast<CBaseFile::TSeekPosition>( 0 ) );
    }
    batchSize.Add( 0, count );
}

CDistributedTraining::CDistributedTraining( CDnn& dnn, int count, TDistributedInitializer initializer, int seed ) :
    physicalCpuCount( getPhysicalCpuCount() )
{
    mathEngines.SetSize( count );
    CreateDistributedCpuMathEngines( mathEngines.GetPtr(), count );
    CMemoryFile file;
    CArchive archive( &file, CArchive::SD_Storing );
    dnn.Serialize( archive );
    archive.Close();
    file.SeekToBegin();

    archive.Open( &file, CArchive::SD_Loading );
    initialize( archive, count, initializer, seed );
    archive.Close();
    file.SeekToBegin();

    archive.Open( &file, CArchive::SD_Storing );
    CPtr<CDnnSolver> solver = dnn.GetSolver();
    SerializeSolver( archive, dnn, solver );
    archive.Close();
    file.SeekToBegin();

    archive.Open( &file, CArchive::SD_Loading );
    SetSolver( archive );
}

CDistributedTraining::CDistributedTraining( CArchive& archive, int count, TDistributedInitializer initializer, int seed ) :
    physicalCpuCount( getPhysicalCpuCount() )
{
    mathEngines.SetSize( count );
    CreateDistributedCpuMathEngines( mathEngines.GetPtr(), count );
    initialize( archive, count, initializer, seed );
}

CDistributedTraining::CDistributedTraining( CDnn& dnn, const CArray<int>& cudaDevs,
    TDistributedInitializer initializer, int seed )
{
    mathEngines.SetSize( cudaDevs.Size() );
    CreateDistributedCudaMathEngines( mathEngines.GetPtr(), cudaDevs.Size(), cudaDevs.GetPtr() );
    CMemoryFile file;
    CArchive archive( &file, CArchive::SD_Storing );
    dnn.Serialize( archive );
    archive.Close();
    file.SeekToBegin();

    archive.Open( &file, CArchive::SD_Loading );
    initialize( archive, cudaDevs.Size(), initializer, seed );
    archive.Close();
    file.SeekToBegin();

    archive.Open( &file, CArchive::SD_Storing );
    CPtr<CDnnSolver> solver = dnn.GetSolver();
    SerializeSolver( archive, dnn, solver );
    archive.Close();
    file.SeekToBegin();

    archive.Open( &file, CArchive::SD_Loading );
    SetSolver( archive );
}

CDistributedTraining::CDistributedTraining( CArchive& archive, const CArray<int>& cudaDevs,
    TDistributedInitializer initializer, int seed )
{
    mathEngines.SetSize( cudaDevs.Size() );
    CreateDistributedCudaMathEngines( mathEngines.GetPtr(), cudaDevs.Size(), cudaDevs.GetPtr() );
    initialize( archive, cudaDevs.Size(), initializer, seed );
}

CDistributedTraining::~CDistributedTraining()
{
    for( int i = 0; i < cnns.Size(); i++ ){
        delete cnns[i];
        delete rands[i];
        delete mathEngines[i];
    }
}

void CDistributedTraining::SetSolver( CArchive& archive )
{
    NeoAssert( archive.IsLoading() );
    long long startPos = archive.GetPosition();
    for( int i = 0; i < cnns.Size(); ++i ) {
        CPtr<CDnnSolver> newSolver = nullptr;
        SerializeSolver( archive, *cnns[i], newSolver );
        cnns[i]->SetSolver( newSolver );
        archive.Seek( startPos, CBaseFile::begin );
    }
}

void CDistributedTraining::SetLearningRate( float newRate )
{
    for( int i = 0; i < cnns.Size(); ++i ) {
        cnns[i]->GetSolver()->SetLearningRate( newRate );
    }
}

void CDistributedTraining::RunOnce( IDistributedDataset& data )
{
#ifdef NEOML_USE_OMP
    NEOML_OMP_NUM_THREADS( cnns.Size() )
    {
        const int thread = OmpGetThreadNum();
        try {
            CThreadGroupSwitcher groupSwitcher( thread, cnns.Size(), physicalCpuCount );
            const int currBatchSize = data.SetInputBatch( *cnns[thread], thread );
            NeoAssert( currBatchSize > 0 || ( currBatchSize == 0 && !isFirstRun ) );
            if( currBatchSize > 0 ) {
                batchSize[thread] += currBatchSize;
                cnns[thread]->RunOnce();
            }
            isFirstRun = false;
        } catch( std::exception& e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e.what();
            }
            cnns[thread]->GetMathEngine().AbortDistributed();
        }
#ifdef NEOML_USE_FINEOBJ
        catch( CCheckException* e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e->MessageText().CreateString();
            }
            cnns[thread]->GetMathEngine().AbortDistributed();
            delete e;
        }
#endif
    }
    CheckArchitecture( errorMessage.IsEmpty(), "DistributedTraining", errorMessage );
#else
    ( void ) data;
    NeoAssert( false );
#endif
}

void CDistributedTraining::RunAndBackwardOnce( IDistributedDataset& data )
{
#ifdef NEOML_USE_OMP
    NEOML_OMP_NUM_THREADS( cnns.Size() )
    {
        const int thread = OmpGetThreadNum();
        try {
            CThreadGroupSwitcher groupSwitcher( thread, cnns.Size(), physicalCpuCount );
            const int currBatchSize = data.SetInputBatch( *cnns[thread], thread );
            NeoAssert( currBatchSize > 0 || ( currBatchSize == 0 && !isFirstRun ) );
            if( currBatchSize > 0 ) {
                batchSize[thread] += currBatchSize;
                cnns[thread]->RunAndBackwardOnce();
            }
            isFirstRun = false;
        } catch( std::exception& e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e.what();
            }
            cnns[thread]->GetMathEngine().AbortDistributed();
        }
#ifdef NEOML_USE_FINEOBJ
        catch( CCheckException* e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e->MessageText().CreateString();
            }
            cnns[thread]->GetMathEngine().AbortDistributed();
            delete e;
        }
#endif
    }
    CheckArchitecture( errorMessage.IsEmpty(), "DistributedTraining", errorMessage );
#else
    ( void ) data;
    NeoAssert( false );
#endif
}

void CDistributedTraining::RunAndLearnOnce( IDistributedDataset& data )
{
    RunAndBackwardOnce( data );
    Train();
}

void CDistributedTraining::Train()
{
    NeoAssert( !isFirstRun );
    int totalBatch = 0;
    for( int i = 0; i < batchSize.Size(); ++i ) {
        totalBatch += batchSize[i];
    }
#ifdef NEOML_USE_OMP
    NEOML_OMP_NUM_THREADS( cnns.Size() )
    {
        const int thread = OmpGetThreadNum();
        try {
            CThreadGroupSwitcher groupSwitcher( thread, cnns.Size(), physicalCpuCount );
            cnns[thread]->GetSolver()->Train( batchSize[thread] * cnns.Size() / static_cast<float>( totalBatch ) );
            batchSize[thread] = 0;
        } catch( std::exception& e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e.what();
            }
            cnns[thread]->GetMathEngine().AbortDistributed();
        }
#ifdef NEOML_USE_FINEOBJ
        catch( CCheckException* e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e->MessageText().CreateString();
            }
            cnns[thread]->GetMathEngine().AbortDistributed();
            delete e;
        }
#endif
    }
    CheckArchitecture( errorMessage.IsEmpty(), "DistributedTraining", errorMessage );
#else
    NeoAssert( false );
#endif
}

void CDistributedTraining::GetLastLoss( const CString& layerName, CArray<float>& losses )
{
    losses.SetSize( cnns.Size() );
    for( int i = 0; i < cnns.Size(); i++ ){
        CLossLayer* lossLayer = dynamic_cast<CLossLayer*>( cnns[i]->GetLayer( layerName ).Ptr() );
        if( lossLayer == nullptr ){
            CCtcLossLayer* ctc = dynamic_cast<CCtcLossLayer*>( cnns[i]->GetLayer( layerName ).Ptr() );
            if( ctc != nullptr ) {
                losses[i] = ctc->GetLastLoss();
            } else {
                losses[i] = CheckCast<CCrfLossLayer>( cnns[i]->GetLayer( layerName ) )->GetLastLoss();
            }
        } else {
            losses[i] = lossLayer->GetLastLoss();
        }
    }
}

void CDistributedTraining::GetLastBlob( const CString& layerName, CObjectArray<CDnnBlob>& blobs )
{
    blobs.SetSize( cnns.Size() );
    for( int i = 0; i < cnns.Size(); ++i ) {
        blobs[i] = CheckCast<CSinkLayer>( cnns[i]->GetLayer( layerName ) )->GetBlob();
    }
}

void CDistributedTraining::Serialize( CArchive& archive )
{
    NeoAssert( archive.IsStoring() );
    archive.Serialize( *cnns[0] );
}

} // namespace NeoML