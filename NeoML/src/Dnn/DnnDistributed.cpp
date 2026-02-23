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

#else  // !FINE_PLATFORM( FINE_WINDOWS )

static void initThreadGroupInfo() {}

#endif // !FINE_PLATFORM( FINE_WINDOWS )

//---------------------------------------------------------------------------------------------------------------------

// RAII switcher of current thread's group
class CThreadGroupSwitcher final {
public:
	CThreadGroupSwitcher( bool isCpu, int threadIndex, int threadCount );
	CThreadGroupSwitcher( const CThreadGroupSwitcher& ) = delete;

	~CThreadGroupSwitcher();

	CThreadGroupSwitcher operator=( const CThreadGroupSwitcher& ) = delete;

private:
#if FINE_PLATFORM( FINE_WINDOWS )
	bool isAffinitySet = false;
	GROUP_AFFINITY prevAffinity{};
#endif // FINE_PLATFORM( FINE_WINDOWS )
};

CThreadGroupSwitcher::CThreadGroupSwitcher( bool isCpu, int threadIndex, int threadCount )
{
#if FINE_PLATFORM( FINE_WINDOWS )
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
#else  // !FINE_PLATFORM( FINE_WINDOWS )
	( void ) isCpu;
	( void ) threadIndex;
	( void ) threadCount;
#endif // !FINE_PLATFORM( FINE_WINDOWS )
}

CThreadGroupSwitcher::~CThreadGroupSwitcher()
{
#if FINE_PLATFORM( FINE_WINDOWS )
	if( isAffinitySet ) {
		setThreadGroupAffinity( ::GetCurrentThread(), &prevAffinity, nullptr );
	}
#endif // FINE_PLATFORM( FINE_WINDOWS )
}

//---------------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------------

// Params to transfer to all threads function
struct CDistributedTraining::CThreadParams final {
	// `Train()` cannot be called if it `isFirstRun`
	// `batchSize` may not be equal 0, if it `isFirstRun` for `RunOnce`, `RunAndBackwardOnce` or `RunAndLearnOnce`.
	bool IsFirstRun = true;
	
	IDistributedDataset* Data = nullptr; // Pointer to data for the inference for all dnns
	const bool IsCpu; // Either multi-threads on a CPU or multi-devices GPU	
	CPointerArray<CDnn>& Dnns; // Separate dnn for each thread	
	CArray<int> BatchSize; // Separate batch size for each dnn (may be empty) in a thread	
	CArray<CString> ErrorMessages; // Containers for errors if it happened	
	bool IsErrorHappened = false; // Flag indicates the error happened	
	TRunType Type; // The type of multi-threaded operation performed	
	int TotalBatch = 0; // The sum of all BatchSizes

	CThreadParams( bool isCpu, CPointerArray<CDnn>& dnns );

	void SetData( IDistributedDataset* data = nullptr, TRunType type = TRunType::Invalid );
	void SetErrorMessage( int threadIndex, CString message );
	void CallRun( int threadIndex );
};

CDistributedTraining::CThreadParams::CThreadParams( bool isCpu, CPointerArray<CDnn>& dnns ) :
	IsCpu( isCpu ),
	Dnns( dnns )
{
	BatchSize.Add( 0, Dnns.Size() );
	ErrorMessages.Add( {}, Dnns.Size() );
}

void CDistributedTraining::CThreadParams::SetData( IDistributedDataset* data, TRunType type )
{
	IsErrorHappened = false;
	Type = type;
	Data = data;
	TotalBatch = 0;
	if( type == TRunType::Train ) {
		NeoAssert( !IsFirstRun );
		for( int i = 0; i < BatchSize.Size(); ++i ) {
			TotalBatch += BatchSize[i];
		}
	}
}

void CDistributedTraining::CThreadParams::SetErrorMessage( int threadIndex, CString message )
{
	IsErrorHappened = true;
	ErrorMessages[threadIndex] = std::move( message );
	// This abort is monitored only inside:
	//   - CDnnSolver::allReduce (MathEngine.AllReduce)
	//   - CDnnDistributedInitializer::InitializeLayerParams (MathEngine.Broadcast)
	Dnns[threadIndex]->GetMathEngine().AbortDistributed();
	// For dnn.RunOnce or dnn.RunBackwardOnce other threads will not stop
}

void CDistributedTraining::CThreadParams::CallRun( int threadIndex )
{
	CThreadGroupSwitcher groupSwitcher( IsCpu, threadIndex, Dnns.Size() );

	if( Type == TRunType::Train ) {
		const float distributedCoeff = BatchSize[threadIndex] * Dnns.Size() / static_cast<float>( TotalBatch );
		Dnns[threadIndex]->GetSolver()->Train( distributedCoeff );
		BatchSize[threadIndex] = 0;
	} else {
		const int currBatchSize = Data->SetInputBatch( *Dnns[threadIndex], threadIndex );
		NeoAssert( currBatchSize > 0 || ( currBatchSize == 0 && !IsFirstRun ) );
		if( currBatchSize > 0 ) {
			BatchSize[threadIndex] += currBatchSize;
			switch( Type ) {
				case TRunType::RunOnce:
					Dnns[threadIndex]->RunOnce();
					break;
				case TRunType::RunBackwardOnce:
					Dnns[threadIndex]->RunAndBackwardOnce();
					break;
				default:
					NeoAssert( false );
			}
		}
		IsFirstRun = false;
	}
}

//---------------------------------------------------------------------------------------------------------------------

void CDistributedTraining::initialize( CArchive& archive, int count,
		TDistributedInitializer initializer, int seed, size_t memoryLimit, const int* cudaDevs )
{
	NeoAssert( archive.IsLoading() );
	mathEngines.SetSize( count );
	if( cudaDevs == nullptr ) {
		initThreadGroupInfo();
		CreateDistributedCpuMathEngines( mathEngines.GetPtr(), count, memoryLimit );
	} else {
		CreateDistributedCudaMathEngines( mathEngines.GetPtr(), count, cudaDevs, memoryLimit );
	}
	rands.SetBufferSize( count );
	cnns.SetBufferSize( count );
	for( int i = 0; i < count; ++i ) {
		rands.Add( new CRandom( seed ) );
		cnns.Add( new CDnn( *rands[i], *mathEngines[i] ) );
		cnns[i]->SetInitializer( createInitializer( initializer, *rands[i] ) );
		cnns[i]->SetInitializer( new CDnnDistributedInitializer( *rands[i], mathEngines[i], cnns[i]->GetInitializer() ) );
		archive.Serialize( *cnns[i] );
		archive.Seek( 0, static_cast<CBaseFile::TSeekPosition>( 0 ) );
	}
}

void CDistributedTraining::serializeDnn( CDnn& dnn, int count,
		TDistributedInitializer initializer, int seed, size_t memoryLimit, const int* cudaDevs )
{
	CMemoryFile file;
	CArchive archive( &file, CArchive::SD_Storing );
	dnn.Serialize( archive );
	archive.Close();
	file.SeekToBegin();

	archive.Open( &file, CArchive::SD_Loading );
	initialize( archive, count, initializer, seed, memoryLimit, cudaDevs );
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

CDistributedTraining::CDistributedTraining( const CDnn& dnn, int count,
		TDistributedInitializer initializer, int seed, size_t memoryLimit ) :
	threadPool( CreateThreadPool( count ) )
{
	// if count was <= 0 the threadPool->Size() has been initialized with the number of available CPU cores
	serializeDnn( const_cast<CDnn&>( dnn ), threadPool->Size(), initializer, seed, memoryLimit );
	threadParams = new CThreadParams( /*isCpu*/true, cnns );
}

CDistributedTraining::CDistributedTraining( CArchive& archive, int count,
		TDistributedInitializer initializer, int seed, size_t memoryLimit ) :
	threadPool( CreateThreadPool( count ) )
{
	// if count was <= 0 the threadPool->Size() has been initialized with the number of available CPU cores
	initialize( archive, threadPool->Size(), initializer, seed, memoryLimit );
	threadParams = new CThreadParams( /*isCpu*/true, cnns );
}

CDistributedTraining::CDistributedTraining( const CDnn& dnn, const CArray<int>& cudaDevs,
		TDistributedInitializer initializer, int seed, size_t memoryLimit ) :
	threadPool( CreateThreadPool( cudaDevs.Size() ) )
{
	serializeDnn( const_cast<CDnn&>( dnn ), cudaDevs.Size(), initializer, seed, memoryLimit, cudaDevs.GetPtr() );
	threadParams = new CThreadParams( /*isCpu*/false, cnns );
}

CDistributedTraining::CDistributedTraining( CArchive& archive, const CArray<int>& cudaDevs,
		TDistributedInitializer initializer, int seed, size_t memoryLimit ) :
	threadPool( CreateThreadPool( cudaDevs.Size() ) )
{
	initialize( archive, cudaDevs.Size(), initializer, seed, memoryLimit, cudaDevs.GetPtr() );
	threadParams = new CThreadParams( /*isCpu*/false, cnns );
}

CDistributedTraining::~CDistributedTraining()
{
	cnns.DeleteAll();
	rands.DeleteAll();
	// As mathEngines are owned, there are no buffers in pools left for any thread
	for( int i = 0; i < mathEngines.Size(); ++i ) {
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

float CDistributedTraining::GetLearningRate() const
{
	return cnns[0]->GetSolver()->GetLearningRate();
}

void CDistributedTraining::run( IDistributedDataset* data, TRunType type )
{
	threadParams->SetData( data, type );

	IThreadPool::TFunction f = []( int threadIndex, void* ptr )
	{
		CThreadParams& threadParams = *static_cast<CThreadParams*>( ptr );
		try {
			threadParams.CallRun( threadIndex );
		} catch( std::exception& e ) {
			threadParams.SetErrorMessage( threadIndex, e.what() );
		}
#ifdef NEOML_USE_FINEOBJ
		catch( CException* e ) {
			threadParams.SetErrorMessage( threadIndex, e->MessageText().CreateString() );
			delete e;
		}
#endif // NEOML_USE_FINEOBJ
	};
	NEOML_NUM_THREADS( *threadPool, threadParams, f );

	CheckArchitecture( !threadParams->IsErrorHappened, "DistributedTraining",
		JoinStrings( threadParams->ErrorMessages ) );
	threadParams->SetData();
}

void CDistributedTraining::RunOnce( IDistributedDataset& data )
{
	run( &data, TRunType::RunOnce );
}

void CDistributedTraining::RunAndBackwardOnce( IDistributedDataset& data )
{
	run( &data, TRunType::RunBackwardOnce );
}

void CDistributedTraining::RunAndLearnOnce( IDistributedDataset& data )
{
	RunAndBackwardOnce( data );
	Train();
}

void CDistributedTraining::Train()
{
	run( nullptr, TRunType::Train );
}

void CDistributedTraining::GetLastLoss( const CString& layerName, CArray<float>& losses ) const
{
	losses.SetSize( cnns.Size() );
	for( int i = 0; i < cnns.Size(); ++i ) {
		const CBaseLayer* layer = cnns[i]->GetLayer( layerName ).Ptr();
		auto lossLayer = dynamic_cast<const CLossLayer*>( layer );
		if( lossLayer != nullptr ) {
			losses[i] = lossLayer->GetLastLoss();
		} else {
			auto ctc = dynamic_cast<const CCtcLossLayer*>( layer );
			if( ctc != nullptr ) {
				losses[i] = ctc->GetLastLoss();
			} else {
				losses[i] = CheckCast<const CCrfLossLayer>( layer )->GetLastLoss();
			}
		}
	}
}

void CDistributedTraining::GetLastBlob( const CString& layerName, CObjectArray<const CDnnBlob>& blobs ) const
{
	blobs.SetSize( cnns.Size() );
	for( int i = 0; i < cnns.Size(); ++i ) {
		blobs[i] = CheckCast<const CSinkLayer>( cnns[i]->GetLayer( layerName ) )->GetBlob();
	}
}

// deprecated
void CDistributedTraining::GetLastBlob( const CString& layerName, CObjectArray<CDnnBlob>& blobs ) const
{
	GetLastBlob( layerName, reinterpret_cast<CObjectArray<const CDnnBlob>&>( blobs ) );
}

void CDistributedTraining::Serialize( CArchive& archive )
{
	// save the first dnn without solver data
	StoreDnn( archive, 0, false );
}

void CDistributedTraining::StoreDnn( CArchive& archive, int index, bool storeSolver )
{
	NeoAssert( archive.IsStoring() );
	NeoAssert( cnns.IsValidIndex( index ) && cnns[index] != nullptr );

	if( storeSolver ) {
		cnns[index]->SerializeCheckpoint( archive );
	} else {
		cnns[index]->Serialize( archive );
	}
}

//---------------------------------------------------------------------------------------------------------------------

// Params to transfer to all threads function
struct CDistributedInference::CThreadParams final {
	IDistributedDataset* Data = nullptr; // Pointer to data for the inference for all dnns
	CObjectArray<CDnnReference> Refs; // Separate dnn for each thread
	CArray<bool> IsDnnInferenced; // Indicates for what dnns the inference was performed
	CArray<CString> ErrorMessages; // Containers for errors if it happened
	bool IsErrorHappened = false; // Flag indicates the error happened

	CThreadParams( int threadsCount, CReferenceDnnFactory& referenceDnnFactory );

	void Initialize( IDistributedDataset& data );
	void DropData() { Data = nullptr; }
	void SetErrorMessage( int threadIndex, CString message );
};

CDistributedInference::CThreadParams::CThreadParams( int threadsCount, CReferenceDnnFactory& referenceDnnFactory )
{
	initThreadGroupInfo();

	// Create reference dnns
	// To create a reference dnn the original network should be trained or at least reshaped
	// All training paramBlobs should exist
	Refs.SetBufferSize( threadsCount );
	for( int i = 1; i < threadsCount; ++i ) {
		Refs.Add( referenceDnnFactory.CreateReferenceDnn() );
	}
	// Here it can be either a one more reference dnn
	// Or also the original dnn, because no one can create a new reference dnn, while this inference
	Refs.Add( referenceDnnFactory.CreateReferenceDnn( /*getOriginalDnn*/true ) );
}

void CDistributedInference::CThreadParams::Initialize( IDistributedDataset& data )
{
	Data = &data;
	IsDnnInferenced.DeleteAll();
	IsDnnInferenced.Add( false, Refs.Size() );
	ErrorMessages.DeleteAll();
	ErrorMessages.Add( CString{}, Refs.Size() );
	IsErrorHappened = false;
}

void CDistributedInference::CThreadParams::SetErrorMessage( int threadIndex, CString message )
{
	IsErrorHappened = true;
	ErrorMessages[threadIndex] = std::move( message );
}

//---------------------------------------------------------------------------------------------------------------------

CDistributedInference::CDistributedInference( const CDnn& dnn, int threadsCount,
		bool optimizeDnn, size_t memoryLimit ) :
	threadPool( CreateThreadPool( threadsCount ) ),
	mathEngine( CreateCpuMathEngine( memoryLimit ) ),
	referenceDnnFactory( new CReferenceDnnFactory( *mathEngine, dnn, optimizeDnn ) ),
	// if count was <= 0 the threadPool->Size() has been initialized with the number of available CPU cores
	threadParams( new CThreadParams( threadPool->Size(), *referenceDnnFactory ) )
{
}

CDistributedInference::CDistributedInference( CArchive& archive, int threadsCount, int seed,
		bool optimizeDnn, size_t memoryLimit ) :
	threadPool( CreateThreadPool( threadsCount ) ),
	mathEngine( CreateCpuMathEngine( memoryLimit ) ),
	referenceDnnFactory( new CReferenceDnnFactory( *mathEngine, archive, seed, optimizeDnn ) ),
	// if count was <= 0 the threadPool->Size() has been initialized with the number of available CPU cores
	threadParams( new CThreadParams( threadPool->Size(), *referenceDnnFactory ) )
{
}

CDistributedInference::~CDistributedInference()
{
}

void CDistributedInference::RunOnce( IDistributedDataset& data )
{
	threadParams->Initialize( data );

	IThreadPool::TFunction f = []( int threadIndex, void* ptr )
	{
		CThreadParams& threadParams = *static_cast<CThreadParams*>( ptr );
		try {
			CThreadGroupSwitcher groupSwitcher( /*isCpu*/true, threadIndex, threadParams.Refs.Size() );
			// Returns the current batch size (or 0, if there is no data for this thread on this run)
			const int currBatchSize = threadParams.Data->SetInputBatch( threadParams.Refs[threadIndex]->Dnn, threadIndex );
			if( currBatchSize > 0 ) { // If thread has some data to perform
				threadParams.Refs[threadIndex]->Dnn.RunOnce();
				threadParams.IsDnnInferenced[threadIndex] = true;
			}
		} catch( std::exception& e ) {
			threadParams.SetErrorMessage( threadIndex, e.what() );
		}
#ifdef NEOML_USE_FINEOBJ
		catch( CException* e ) {
			threadParams.SetErrorMessage( threadIndex, e->MessageText().CreateString() );
			delete e;
		}
#endif // NEOML_USE_FINEOBJ
	};
	NEOML_NUM_THREADS( *threadPool, threadParams, f );

	CheckArchitecture( !threadParams->IsErrorHappened, "DistributedTraining",
		JoinStrings( threadParams->ErrorMessages ) );
	threadParams->DropData();
}

void CDistributedInference::GetLastBlob( const CString& layerName, CObjectArray<const CDnnBlob>& blobs ) const
{
	blobs.DeleteAll();
	blobs.Add( nullptr, threadParams->Refs.Size() );
	for( int i = 0; i < threadParams->Refs.Size(); ++i ) {
		if( threadParams->IsDnnInferenced[i] ) {
			blobs[i] = CheckCast<const CSinkLayer>( threadParams->Refs[i]->Dnn.GetLayer( layerName ) )->GetBlob();
		}
	}
}

} // namespace NeoML
