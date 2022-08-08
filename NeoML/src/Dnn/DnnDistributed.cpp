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
#include <NeoMathEngine/ThreadPool.h>

#if FINE_PLATFORM( FINE_WINDOWS )
#include <Windows.h>
#endif

namespace NeoML {

#if FINE_PLATFORM( FINE_WINDOWS )

typedef BOOL (WINAPI *TGetProcessGroupAffinity) (HANDLE, PUSHORT, PUSHORT);
typedef BOOL (WINAPI *TSetThreadGroupAffinity) (HANDLE, const GROUP_AFFINITY*, PGROUP_AFFINITY);
typedef BOOL (WINAPI *TGetThreadGroupAffinity) (HANDLE, PGROUP_AFFINITY);

static TGetProcessGroupAffinity getProcessGroupAffinity = nullptr;
static TGetThreadGroupAffinity getThreadGroupAffinity = nullptr;
static TSetThreadGroupAffinity setThreadGroupAffinity = nullptr;

static const int threadGroupMax = 32;
static USHORT threadGroupCount = 1;
static USHORT threadGroups[threadGroupMax];

static bool isWinApiPartInitialized = false;
static CCriticalSection functionLoadSection;

void initThreadGroupInfo()
{
    CCriticalSectionLock lock( functionLoadSection );
    if( isWinApiPartInitialized ) {
        return;
    }

    isWinApiPartInitialized = true;

    HMODULE kernel32Handle = ::GetModuleHandle( TEXT( "kernel32" ) );
    if( kernel32Handle == NULL ) {
        return;
    }

    // resets static winAPI part in case of failure
    auto onFailure = [] () {
        threadGroupCount = 1;
        getProcessGroupAffinity = nullptr;
        getThreadGroupAffinity = nullptr;
        setThreadGroupAffinity = nullptr;
    };

    getProcessGroupAffinity = ( TGetProcessGroupAffinity ) ::GetProcAddress( kernel32Handle, "GetProcessGroupAffinity" );
    getThreadGroupAffinity = ( TGetThreadGroupAffinity ) ::GetProcAddress( kernel32Handle, "GetThreadGroupAffinity" );
    setThreadGroupAffinity = ( TSetThreadGroupAffinity ) ::GetProcAddress( kernel32Handle, "SetThreadGroupAffinity" );

    if( getProcessGroupAffinity == nullptr || getThreadGroupAffinity == nullptr || setThreadGroupAffinity == nullptr ) {
        onFailure();
        return;
    }

    threadGroupCount = threadGroupMax;
    if( getProcessGroupAffinity( ::GetCurrentProcess(), &threadGroupCount, threadGroups ) == 0 ) {
        onFailure();
        return;
    }
}

#else // FINE_PLATFORM( FINE_WINDOWS )

static void initThreadGroupInfo() {}

#endif

// RAII switcher of current thread's group
class CThreadGroupSwitcher {
public:
#if FINE_PLATFORM( FINE_WINDOWS )
    CThreadGroupSwitcher( bool isCpu, int threadIndex, int threadCount ) :
        isAffinitySet( false )
    {
        if( isCpu && threadGroupCount > 1 ) {
            // spread threads equally between multiple thread groups
            const int threadPerGroup = ( threadCount + threadGroupCount - 1 ) / threadGroupCount;
            GROUP_AFFINITY affinity;
            affinity.Reserved[0] = 0;
            affinity.Reserved[1] = 0;
            affinity.Reserved[2] = 0;
            NeoAssert( getThreadGroupAffinity != nullptr );
            if( getThreadGroupAffinity( ::GetCurrentThread(), &affinity ) != 0 ) {
                affinity.Group = threadGroups[threadIndex / threadPerGroup];
                isAffinitySet = ( setThreadGroupAffinity( ::GetCurrentThread(), &affinity, &prevAffinity ) != 0 );
            }
        }
    }

    ~CThreadGroupSwitcher()
    {
        if( isAffinitySet ) {
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
    bool isAffinitySet;
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
    isCpu( true ),
    threadPool( CreateThreadPool(count) )
{
    initThreadGroupInfo();
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
    isCpu( true ),
    threadPool( CreateThreadPool(count) )
{
    initThreadGroupInfo();
    mathEngines.SetSize( count );
    CreateDistributedCpuMathEngines( mathEngines.GetPtr(), count );
    initialize( archive, count, initializer, seed );
}

CDistributedTraining::CDistributedTraining( CDnn& dnn, const CArray<int>& cudaDevs,
        TDistributedInitializer initializer, int seed ) :
    isCpu( false ),
    threadPool( CreateThreadPool(cudaDevs.Size()) )
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
        TDistributedInitializer initializer, int seed ) :
    isCpu( false ),
    threadPool( CreateThreadPool(cudaDevs.Size()) )
{
    mathEngines.SetSize( cudaDevs.Size() );
    CreateDistributedCudaMathEngines( mathEngines.GetPtr(), cudaDevs.Size(), cudaDevs.GetPtr() );
    initialize( archive, cudaDevs.Size(), initializer, seed );
}

CDistributedTraining::~CDistributedTraining()
{
    delete threadPool;
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
    struct CFunctionParams {
        bool& IsFirstRun;
        IDistributedDataset& Data;
        CArray<CDnn*>& Cnns;
        CArray<int>& BatchSize;
        bool IsCpu;
        CString& ErrorMessage;

        CFunctionParams(bool& isFirstRun, IDistributedDataset& data, CArray<CDnn*>& cnns, CArray<int>& batchSize, bool isCpu, CString& errorMessage) :
            IsFirstRun(isFirstRun),
            Data(data),
            Cnns(cnns),
            BatchSize(batchSize),
            IsCpu(isCpu),
            ErrorMessage(errorMessage)
        {
        }
    } function_params(isFirstRun, data, cnns, batchSize, isCpu, errorMessage);

    IThreadPool::TFunction f = [](int threadIndex, void* ptr)
    {
        CFunctionParams& function_params = *(CFunctionParams*)ptr;
        CArray<CDnn*>& cnns = function_params.Cnns;
        CArray<int>& batchSize = function_params.BatchSize;
        CString& errorMessage = function_params.ErrorMessage;
        try {
            CThreadGroupSwitcher groupSwitcher( function_params.IsCpu, threadIndex, cnns.Size() );
            const int currBatchSize = function_params.Data.SetInputBatch( *cnns[threadIndex], threadIndex );
            NeoAssert( currBatchSize > 0 || ( currBatchSize == 0 && !function_params.IsFirstRun ) );
            if( currBatchSize > 0 ) {
                batchSize[threadIndex] += currBatchSize;
                cnns[threadIndex]->RunOnce();
            }
            function_params.IsFirstRun = false;
        } catch( std::exception& e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e.what();
            }
            cnns[threadIndex]->GetMathEngine().AbortDistributed();
        }
#ifdef NEOML_USE_FINEOBJ
        catch( CCheckException* e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e->MessageText().CreateString();
            }
            cnns[threadIndex]->GetMathEngine().AbortDistributed();
            delete e;
        }
#endif
    };
    NEOML_NUM_THREADS( *threadPool, &function_params, f );

    CheckArchitecture( errorMessage.IsEmpty(), "DistributedTraining", errorMessage );
}

void CDistributedTraining::RunAndBackwardOnce( IDistributedDataset& data )
{
    struct CFunctionParams {
        bool& IsFirstRun;
        IDistributedDataset& Data;
        CArray<CDnn*>& Cnns;
        CArray<int>& BatchSize;
        bool IsCpu;
        CString& ErrorMessage;

        CFunctionParams(bool& isFirstRun, IDistributedDataset& data, CArray<CDnn*>& cnns, CArray<int>& batchSize, bool isCpu, CString& errorMessage) :
            IsFirstRun(isFirstRun),
            Data(data),
            Cnns(cnns),
            BatchSize(batchSize),
            IsCpu(isCpu),
            ErrorMessage(errorMessage)
        {
        }
    } function_params(isFirstRun, data, cnns, batchSize, isCpu, errorMessage);

    IThreadPool::TFunction f = [](int threadIndex, void* ptr)
    {
        CFunctionParams& function_params = *(CFunctionParams*)ptr;
        CArray<CDnn*>& cnns = function_params.Cnns;
        CArray<int>& batchSize = function_params.BatchSize;
        CString& errorMessage = function_params.ErrorMessage;
        try {
            CThreadGroupSwitcher groupSwitcher( function_params.IsCpu, threadIndex, cnns.Size() );
            const int currBatchSize = function_params.Data.SetInputBatch( *cnns[threadIndex], threadIndex );
            NeoAssert( currBatchSize > 0 || ( currBatchSize == 0 && !function_params.IsFirstRun ) );
            if( currBatchSize > 0 ) {
                batchSize[threadIndex] += currBatchSize;
                cnns[threadIndex]->RunAndBackwardOnce();
            }
            function_params.IsFirstRun = false;
        } catch( std::exception& e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e.what();
            }
            cnns[threadIndex]->GetMathEngine().AbortDistributed();
        }
#ifdef NEOML_USE_FINEOBJ
        catch( CCheckException* e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e->MessageText().CreateString();
            }
            cnns[threadIndex]->GetMathEngine().AbortDistributed();
            delete e;
        }
#endif
    };
    NEOML_NUM_THREADS( *threadPool, &function_params, f );

    CheckArchitecture( errorMessage.IsEmpty(), "DistributedTraining", errorMessage );
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

    struct CFunctionParams {
        CArray<CDnn*>& Cnns;
        CArray<int>& BatchSize;
        int TotalBatch;
        bool IsCpu;
        CString& ErrorMessage;

        CFunctionParams(CArray<CDnn*>& cnns, CArray<int>& batchSize, int totalBatch, bool isCpu, CString& errorMessage) :
            Cnns(cnns),
            BatchSize(batchSize),
            TotalBatch(totalBatch),
            IsCpu(isCpu),
            ErrorMessage(errorMessage)
        {
        }
    } function_params(cnns, batchSize, totalBatch, isCpu, errorMessage);

    IThreadPool::TFunction f = [](int threadIndex, void* ptr)
    {
        CFunctionParams& function_params = *(CFunctionParams*)ptr;
        CArray<CDnn*>& cnns = function_params.Cnns;
        CArray<int>& batchSize = function_params.BatchSize;
        CString& errorMessage = function_params.ErrorMessage;

        try {
            CThreadGroupSwitcher groupSwitcher( function_params.IsCpu, threadIndex, cnns.Size() );
            cnns[threadIndex]->GetSolver()->Train( batchSize[threadIndex] * cnns.Size() / static_cast<float>( function_params.TotalBatch ) );
            batchSize[threadIndex] = 0;
        } catch( std::exception& e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e.what();
            }
            cnns[threadIndex]->GetMathEngine().AbortDistributed();
        }
#ifdef NEOML_USE_FINEOBJ
        catch( CCheckException* e ) {
            if( errorMessage.IsEmpty() ) {
                errorMessage = e->MessageText().CreateString();
            }
            cnns[threadIndex]->GetMathEngine().AbortDistributed();
            delete e;
        }
#endif
    };
    NEOML_NUM_THREADS( *threadPool, &function_params, f );

    CheckArchitecture( errorMessage.IsEmpty(), "DistributedTraining", errorMessage );
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
