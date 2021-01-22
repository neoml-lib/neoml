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

#include <NeoProxy/NeoProxy.h>

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/DnnBlob.h>
#include <NeoML/ArchiveFile.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoOnnx/NeoOnnx.h>

#include <cstdio>

using namespace NeoML;

extern "C" {

static void initErrorInfo( TDnnErrorType errorType, const char* errorText, struct CDnnErrorInfo* errorInfo )
{
	if( errorInfo == 0 ) {
		return;
	}

	errorInfo->Type = errorType;
	const size_t len = min( sizeof( errorInfo->Description ) - 1, strlen( errorText ) );
	memcpy( errorInfo->Description, errorText, len );
	errorInfo->Description[len] = 0;
}

//------------------------------------------------------------------------------------------------------------

// The file implementation over a buffer
class CBufferFile : public CBaseFile {
public:
	CBufferFile( const void* buffer, __int64 bufferSize );
	virtual ~CBufferFile() = default;

	// CBaseFile class methods
#ifdef FINEOBJ_VERSION
	CUnicodeString GetFileName() const override { return CUnicodeString(L"Buffer"); }
#else
	const char* GetFileName() const override { return "Buffer"; }
#endif
	int Read( void*, int bytesCount ) override;
	void Write( const void*, int ) override { NeoAssert( false ); }
	__int64 GetPosition() const override { NeoAssert( isOpen ); return pos; }
	__int64 Seek( __int64 offset, TSeekPosition from ) override;
	void SetLength( __int64 ) override { NeoAssert( false ); }
	__int64 GetLength() const override { NeoAssert( isOpen ); return bufferSize; }
	void Abort() override;
	void Flush() override { NeoAssert( isOpen ); }
	void Close() override { return Abort(); }

private:
	const void* buffer;
	const __int64 bufferSize;

	bool isOpen;
	__int64 pos;
};

CBufferFile::CBufferFile( const void* _buffer, __int64 _bufferSize ) :
	buffer( _buffer ),
	bufferSize( _bufferSize ),
	isOpen( true ),
	pos( 0 )
{
}

int CBufferFile::Read( void* result, int bytesCount )
{
	NeoAssert( isOpen );
	if( pos >= bufferSize ) {
		return 0;
	}

	const int len = min( bytesCount, static_cast<int>( bufferSize - pos ) );
	::memcpy( result, static_cast<const char*>(buffer) + pos, len );
	pos += len;
	return len;
}

__int64 CBufferFile::Seek( __int64 offset, TSeekPosition from )
{
	NeoAssert( isOpen );
	switch( from ) {
		case begin:
			pos = offset;
			return pos;
		case current:
			pos = min( pos + offset, bufferSize );
			return pos;
		case end:
			pos = max( 0ll, bufferSize - offset );
			return pos;
		default:
			NeoAssert( false );
	};
	return 0;
}

void CBufferFile::Abort()
{
	pos = 0;
	isOpen = false;
}

//------------------------------------------------------------------------------------------------------------
// CDnnMathEngineDesc implementation

class CMathEngineOwner : public IObject {
public:
	explicit CMathEngineOwner( IMathEngine* _mathEngine ) : mathEngine( _mathEngine ) {}

	IMathEngine& MathEngine() const { return *mathEngine; }

protected:
	virtual ~CMathEngineOwner() { delete mathEngine; }

private:
	IMathEngine* mathEngine;
};

//------------------------------------------------------------------------------------------------------------

struct CDnnMathEngineDescImpl : public CDnnMathEngineDesc {
	CPtr<CMathEngineOwner> MathEngineOwner;

	CDnnMathEngineDescImpl( IMathEngine* mathEngine, TDnnMathEngineType engineType );
};

CDnnMathEngineDescImpl::CDnnMathEngineDescImpl( IMathEngine* mathEngine, TDnnMathEngineType engineType ) :
	MathEngineOwner( FINE_DEBUG_NEW CMathEngineOwner( mathEngine ) )
{
	CDnnMathEngineDesc::Type = engineType;
}

//------------------------------------------------------------------------------------------------------------
// MathEngine functions

const struct CDnnMathEngineDesc* CreateGPUMathEngine( struct CDnnErrorInfo* errorInfo )
{
	IMathEngine* mathEngine = nullptr;
	
	try {
		SetMathEngineExceptionHandler( GetExceptionHandler() );
		mathEngine = CreateGpuMathEngine( 0 );

		if( mathEngine == 0 ) {
			initErrorInfo( DET_NoAvailableGPU, "There is no available GPU.", errorInfo );
			return nullptr;
		}

		return FINE_DEBUG_NEW CDnnMathEngineDescImpl( mathEngine, MET_GPU );
#ifdef NEOML_USE_FINEOBJ
	} catch( CException* e ) {
		initErrorInfo( DET_InternalError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
		if( mathEngine != 0 ) {
			delete mathEngine;
		}
	}
#else
	} catch( std::exception& e ) {
		initErrorInfo( DET_InternalError, e.what(), errorInfo );
		if( mathEngine != 0 ) {
			delete mathEngine;
		}
	}
#endif

	return nullptr;
}

const struct CDnnMathEngineDesc* CreateCPUMathEngine( int threadCount, struct CDnnErrorInfo* errorInfo )
{
	IMathEngine* mathEngine = nullptr;
	
	try {
		SetMathEngineExceptionHandler( GetExceptionHandler() );
		mathEngine = CreateCpuMathEngine( threadCount, 0 );

		if( mathEngine == 0 ) {
			initErrorInfo( DET_NoAvailableGPU, "There is no available CPU.", errorInfo );
			return nullptr;
		}

		return FINE_DEBUG_NEW CDnnMathEngineDescImpl( mathEngine, MET_CPU );
#ifdef NEOML_USE_FINEOBJ
	} catch( CException* e ) {
		initErrorInfo( DET_InternalError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
		if( mathEngine != 0 ) {
			delete mathEngine;
		}
	}
#else
	} catch( std::exception& e ) {
		initErrorInfo( DET_InternalError, e.what(), errorInfo );
		if( mathEngine != 0 ) {
			delete mathEngine;
		}
	}
#endif

	return nullptr;
}

void DestroyMathEngine( const struct CDnnMathEngineDesc* mathEngine )
{
	delete static_cast<const CDnnMathEngineDescImpl*>( mathEngine );
}

//------------------------------------------------------------------------------------------------------------
// CDnnBlobDesc implementation

struct CDnnBlobDescImpl : public CDnnBlobDesc {
	CPtr<CDnnBlob> Blob;
	CPtr<CMathEngineOwner> MathEngineOwner;

	CDnnBlobDescImpl( CDnnBlob* blob, const CDnnMathEngineDescImpl* mathEngine ) :
		Blob( blob ),
		MathEngineOwner( mathEngine->MathEngineOwner )
	{
		CDnnBlobDesc::MathEngine = mathEngine;
		CDnnBlobDesc::Type = (TDnnBlobType)blob->GetDataType();
		CDnnBlobDesc::BatchLength = blob->GetBatchLength();
		CDnnBlobDesc::BatchWidth = blob->GetBatchWidth();
		CDnnBlobDesc::Height = blob->GetHeight();
		CDnnBlobDesc::Width = blob->GetWidth();
		CDnnBlobDesc::Depth = blob->GetDepth();
		CDnnBlobDesc::ChannelCount = blob->GetChannelsCount();
		CDnnBlobDesc::DataSize = blob->GetDataSize() * 4;
	}
};

//------------------------------------------------------------------------------------------------------------
// Blob functions

const struct CDnnBlobDesc* CreateDnnBlob( const struct CDnnMathEngineDesc* mathEngineDesc, TDnnBlobType dnnBlobType,
	int batchLength, int batchWidth, int height, int width, int depth, int channelCount, struct CDnnErrorInfo* errorInfo )
{
	const int blobMaxSize = 1024 * 1024 * 1024; // 1GB

	if( dnnBlobType != DBT_Float && dnnBlobType != DBT_Int ) {
		initErrorInfo( DET_InvalidParameter, "Invalid dnnBlobType parameter.", errorInfo );
		return nullptr;
	}
	if( batchLength <= 0 || batchLength > blobMaxSize ) {
		initErrorInfo( DET_InvalidParameter, "Invalid batchLength parameter.", errorInfo );
		return nullptr;
	}
	if( batchWidth <= 0 || batchWidth > blobMaxSize ) {
		initErrorInfo( DET_InvalidParameter, "Invalid batchWidth parameter.", errorInfo );
		return nullptr;
	}
	if( height <= 0 || height > blobMaxSize ) {
		initErrorInfo( DET_InvalidParameter, "Invalid height parameter.", errorInfo );
		return nullptr;
	}
	if( width <= 0 || width > blobMaxSize ) {
		initErrorInfo( DET_InvalidParameter, "Invalid width parameter.", errorInfo );
		return nullptr;
	}
	if( depth <= 0 || depth > blobMaxSize ) {
		initErrorInfo( DET_InvalidParameter, "Invalid depth parameter.", errorInfo );
		return nullptr;
	}
	if( channelCount <= 0 || channelCount > blobMaxSize ) {
		initErrorInfo( DET_InvalidParameter, "Invalid channelCount parameter.", errorInfo );
		return nullptr;
	}
	long long temp[6] = { batchLength, batchWidth, height, width, depth, channelCount };
	long long totalSize = temp[0];
	for( int i = 1; i < 6; i++ ) {
		totalSize *= temp[i];
		if( totalSize > blobMaxSize ) {
			initErrorInfo( DET_InvalidParameter, "Blob size must be smaller than 512Mb.", errorInfo );
			return nullptr;
		}
	}

	const struct CDnnMathEngineDescImpl* mathEngineDescImpl = static_cast<const struct CDnnMathEngineDescImpl*>( mathEngineDesc );

	if( mathEngineDescImpl == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnMathEngineDesc parameter.", errorInfo );
		return nullptr;
	}

	CPtr<CDnnBlob> blob = nullptr;

	try {
		blob = CDnnBlob::Create3DImageBlob( mathEngineDescImpl->MathEngineOwner->MathEngine(), (TBlobType)dnnBlobType,
			batchLength, batchWidth, height, width, depth, channelCount );
		return FINE_DEBUG_NEW CDnnBlobDescImpl( blob, mathEngineDescImpl );
#ifdef NEOML_USE_FINEOBJ
	} catch( CException* e ) {
		initErrorInfo( DET_InternalError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
	}
#else
	} catch( std::exception& e ) {
		initErrorInfo( DET_InternalError, e.what(), errorInfo );
	}
#endif

	return nullptr;
}

void DestroyDnnBlob( const struct CDnnBlobDesc* blob )
{
	delete static_cast<const CDnnBlobDescImpl*>( blob );
}

bool CopyToBlob( const struct CDnnBlobDesc* blobDesc, const void* buffer, struct CDnnErrorInfo* errorInfo )
{
	if( blobDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnBlobDesc parameter.", errorInfo );
		return false;
	}

	static_assert( sizeof( float ) == sizeof( int ), "C API copy functions won't work" );
	const CPtr<CDnnBlob>& blob = static_cast<const struct CDnnBlobDescImpl*>( blobDesc )->Blob;

	if( blob->GetDataType() == CT_Float ) {
		blob->GetMathEngine().DataExchangeRaw( blob->GetData<float>(), buffer, blob->GetDataSize() * sizeof(float) );
	} else if( blob->GetDataType() == CT_Int ) {
		blob->GetMathEngine().DataExchangeRaw( blob->GetData<int>(), buffer, blob->GetDataSize() * sizeof(int) );
	} else {
		initErrorInfo( DET_InternalError, "Wrong data type.", errorInfo );
		return false;
	}
	return true;
}

bool CopyFromBlob( void* buffer, const struct CDnnBlobDesc* blobDesc, struct CDnnErrorInfo* errorInfo )
{
	if( blobDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnBlobDesc parameter.", errorInfo );
		return false;
	}

	static_assert( sizeof( float ) == sizeof( int ), "C API copy functions won't work" );
	const CPtr<CDnnBlob>& blob = static_cast<const struct CDnnBlobDescImpl*>( blobDesc )->Blob;
	if( blob->GetDataType() == CT_Float ) {
		blob->GetMathEngine().DataExchangeRaw( buffer, blob->GetData<float>(), blob->GetDataSize() * sizeof(float) );
	} else if( blob->GetDataType() == CT_Int ) {
		blob->GetMathEngine().DataExchangeRaw( buffer, blob->GetData<int>(), blob->GetDataSize() * sizeof(int) );
	} else {
		initErrorInfo( DET_InternalError, "Wrong data type.", errorInfo );
		return false;
	}
	return true;
}

//------------------------------------------------------------------------------------------------------------
// CDnnDesc implementation

class CDnnDescImpl : public CDnnDesc {
public:
	explicit CDnnDescImpl( const CDnnMathEngineDescImpl* mathEngineDesc );
	CDnnDescImpl( const char* fileName, const CDnnMathEngineDescImpl* mathEngineDesc );
	CDnnDescImpl( const void* buffer, int bufferSize, const CDnnMathEngineDescImpl* mathEngineDesc );

	CDnn& Dnn() { return dnn; }
	const CDnn& Dnn() const { return dnn; }

	void BuildNameList();

	const char* GetInputName( int index ) const { return inputNames[index]; }
	void SetInputBlob( int index, CDnnBlob* blob ) const;

	bool RunOnce( struct CDnnErrorInfo* errorInfo ) const;

	int GetOutputCount() const { return outputNames.Size(); }
	const char* GetOutputName( int index ) const { return outputNames[index]; }
	CPtr<CDnnBlob> GetOutputBlob( int index ) const;

private:
	const CPtr<CMathEngineOwner> mathEngineOwner;
	CRandom random;
	mutable CDnn dnn;

	CArray<CString> inputNames;
	CArray<CString> outputNames;
};

CDnnDescImpl::CDnnDescImpl( const CDnnMathEngineDescImpl* mathEngineDesc ) :
	mathEngineOwner( mathEngineDesc->MathEngineOwner ),
	random( 0x777 ),
	dnn( random, mathEngineDesc->MathEngineOwner->MathEngine() )
{
	CDnnDesc::MathEngine = mathEngineDesc;
}

CDnnDescImpl::CDnnDescImpl( const char* fileName, const CDnnMathEngineDescImpl* mathEngineDesc ) :
	mathEngineOwner( mathEngineDesc->MathEngineOwner ),
	random( 0x777 ),
	dnn( random, mathEngineDesc->MathEngineOwner->MathEngine() )
{
	CDnnDesc::MathEngine = mathEngineDesc;
	{
		CArchiveFile archiveFile( fileName, CArchive::load );
		CArchive archive( &archiveFile, CArchive::SD_Loading );
		dnn.Serialize( archive );
		archive.Close();
		archiveFile.Close();
	}

	BuildNameList();
}

CDnnDescImpl::CDnnDescImpl( const void* buffer, int bufferSize, const CDnnMathEngineDescImpl* mathEngineDesc ) :
	mathEngineOwner( mathEngineDesc->MathEngineOwner ),
	random( 0x777 ),
	dnn( random, mathEngineDesc->MathEngineOwner->MathEngine() )
{
	CDnnDesc::MathEngine = mathEngineDesc;
	{
		CBufferFile bufferFile( buffer, bufferSize );
		CArchive archive( &bufferFile, CArchive::SD_Loading );
		dnn.Serialize( archive );
		archive.Close();
		bufferFile.Close();
	}

	BuildNameList();
}

void CDnnDescImpl::BuildNameList()
{
	CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );

	for( int layerIndex = 0; layerIndex < layerNames.Size(); ++layerIndex ) {
		CPtr<CBaseLayer> layer = dnn.GetLayer( layerNames[layerIndex] );
		if( dynamic_cast<CSourceLayer*>( layer.Ptr() ) != 0 ) {
			inputNames.Add( layer->GetName() );
		} else if( dynamic_cast<CSinkLayer*>( layer.Ptr() ) != 0 ) {
			outputNames.Add( layer->GetName() );
		}
	}
	CDnnDesc::InputCount = inputNames.Size();
	CDnnDesc::OutputCount = outputNames.Size();
}

void CDnnDescImpl::SetInputBlob( int index, CDnnBlob* blob ) const
{
	CSourceLayer* source = dynamic_cast<CSourceLayer*>( dnn.GetLayer( inputNames[index] ).Ptr() );
	source->SetBlob( blob );
}

bool CDnnDescImpl::RunOnce( struct CDnnErrorInfo* errorInfo ) const
{
	try {
		dnn.RunOnce();
		return true;
#ifdef NEOML_USE_FINEOBJ
	} catch( CException* e ) {
		initErrorInfo( DET_RunDnnError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
	}
#else
	} catch( std::exception& e ) {
		initErrorInfo( DET_RunDnnError, e.what(), errorInfo );
	}
#endif
	return false;
}

CPtr<CDnnBlob> CDnnDescImpl::GetOutputBlob( int index ) const
{
	const CSinkLayer* source = dynamic_cast<const CSinkLayer*>( dnn.GetLayer( outputNames[index] ).Ptr() );
	if( source == 0 ) {
		return 0;
	}
	return source->GetBlob();
}

//------------------------------------------------------------------------------------------------------------
// Network functions

const struct CDnnDesc* CreateDnnFromFile( const struct CDnnMathEngineDesc* mathEngineDesc, const char* fileName, struct CDnnErrorInfo* errorInfo )
{
	if( mathEngineDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnMathEngineDesc parameter.", errorInfo );
		return nullptr;
	}

	const CDnnMathEngineDescImpl* mathEngine = static_cast<const CDnnMathEngineDescImpl*>( mathEngineDesc );
	try {
		return FINE_DEBUG_NEW CDnnDescImpl( fileName, mathEngine );
#ifdef NEOML_USE_FINEOBJ
	} catch( CFileException* e ) {
		initErrorInfo( DET_LoadDnnError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
	} catch( CException* e ) {
		initErrorInfo( DET_InternalError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
	}
#else
	} catch( std::system_error& e ) {
		initErrorInfo( DET_LoadDnnError, e.what(), errorInfo );
	} catch( std::exception& e ) {
		initErrorInfo( DET_InternalError, e.what(), errorInfo );
	}
#endif
	return nullptr;
}

const struct CDnnDesc* CreateDnnFromBuffer( const struct CDnnMathEngineDesc* mathEngineDesc, const void* buffer, int bufferSize, struct CDnnErrorInfo* errorInfo )
{
	if( mathEngineDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnMathEngineDesc parameter.", errorInfo );
		return nullptr;
	}
	
	const CDnnMathEngineDescImpl* mathEngine = static_cast<const CDnnMathEngineDescImpl*>( mathEngineDesc );
	try {
		return FINE_DEBUG_NEW CDnnDescImpl( buffer, bufferSize, mathEngine );
#ifdef NEOML_USE_FINEOBJ
	} catch( CFileException* e ) {
		initErrorInfo( DET_LoadDnnError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
	} catch( CException* e ) {
		initErrorInfo( DET_InternalError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
	}
#else
	} catch( std::system_error& e ) {
		initErrorInfo( DET_LoadDnnError, e.what(), errorInfo );
	} catch( std::exception& e ) {
		initErrorInfo( DET_InternalError, e.what(), errorInfo );
	}
#endif
	return nullptr;
}

const struct CDnnDesc* CreateDnnFromOnnxFile( const struct CDnnMathEngineDesc* mathEngineDesc, const char* fileName, struct CDnnErrorInfo* errorInfo )
{
	if( mathEngineDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnMathEngineDesc parameter.", errorInfo );
		return nullptr;
	}

	const CDnnMathEngineDescImpl* mathEngine = static_cast<const CDnnMathEngineDescImpl*>( mathEngineDesc );

	CDnnDescImpl* dnnDesc = nullptr;
	try {
		dnnDesc = FINE_DEBUG_NEW CDnnDescImpl( mathEngine );
		NeoOnnx::LoadFromOnnx( fileName, dnnDesc->Dnn() );
		dnnDesc->BuildNameList();
		return dnnDesc;
#ifdef NEOML_USE_FINEOBJ
	} catch( CException* e ) {
		initErrorInfo( DET_InternalError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
		if( dnnDesc != nullptr ) {
			delete dnnDesc;
		}
	}
#else
	} catch( std::system_error& e ) {
		initErrorInfo( DET_LoadDnnError, e.what(), errorInfo );
		if( dnnDesc != nullptr ) {
			delete dnnDesc;
		}
	} catch( std::exception& e ) {
		initErrorInfo( DET_InternalError, e.what(), errorInfo );
		if( dnnDesc != nullptr ) {
			delete dnnDesc;
		}
	}
#endif
	return nullptr;
}

const struct CDnnDesc* CreateDnnFromOnnxBuffer( const struct CDnnMathEngineDesc* mathEngineDesc, const void* buffer, int bufferSize, struct CDnnErrorInfo* errorInfo )
{
	if( mathEngineDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnMathEngineDesc parameter.", errorInfo );
		return nullptr;
	}

	const CDnnMathEngineDescImpl* mathEngine = static_cast<const CDnnMathEngineDescImpl*>( mathEngineDesc );

	CDnnDescImpl* dnnDesc = nullptr;
	try {
		dnnDesc = FINE_DEBUG_NEW CDnnDescImpl( mathEngine );
		NeoOnnx::LoadFromOnnx( buffer, bufferSize, dnnDesc->Dnn() );
		dnnDesc->BuildNameList();
		return dnnDesc;
#ifdef NEOML_USE_FINEOBJ
	} catch( CException* e ) {
		initErrorInfo( DET_InternalError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
		if( dnnDesc != nullptr ) {
			delete dnnDesc;
		}
	}
#else
	} catch( std::system_error& e ) {
		initErrorInfo( DET_LoadDnnError, e.what(), errorInfo );
		if( dnnDesc != nullptr ) {
			delete dnnDesc;
		}
	} catch( std::exception& e ) {
		initErrorInfo( DET_InternalError, e.what(), errorInfo );
		if( dnnDesc != nullptr ) {
			delete dnnDesc;
		}
	}
#endif
	return nullptr;
}

void DestroyDnn( const struct CDnnDesc* dnnDesc )
{
	delete static_cast<const CDnnDescImpl*>( dnnDesc );
}

const char* GetInputName( const struct CDnnDesc* dnnDesc, int index, struct CDnnErrorInfo* errorInfo )
{
	if( dnnDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnDesc parameter.", errorInfo );
		return 0;
	}

	const CDnnDescImpl* dnn = static_cast<const CDnnDescImpl*>( dnnDesc );
	if( index < 0 || index >= dnn->InputCount ) {
		initErrorInfo( DET_InvalidParameter, "Invalid index.", errorInfo );
		return 0;
	}

	return dnn->GetInputName( index );
}

bool SetInputBlob( const struct CDnnDesc* dnnDesc, int index, const struct CDnnBlobDesc* blobDesc, struct CDnnErrorInfo* errorInfo )
{
	if( dnnDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnDesc parameter.", errorInfo );
		return false;
	}
	if( blobDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnBlobDesc parameter.", errorInfo );
		return false;
	}

	const CDnnDescImpl* dnn = static_cast<const CDnnDescImpl*>( dnnDesc );
	const CDnnBlobDescImpl* blob = static_cast<const CDnnBlobDescImpl*>( blobDesc );

	if( index < 0 || index >= dnn->InputCount ) {
		initErrorInfo( DET_InvalidParameter, "Invalid index.", errorInfo );
		return false;
	}

	dnn->SetInputBlob( index, blob->Blob.Ptr() );
	return true;
}

bool DnnRunOnce( const struct CDnnDesc* dnnDesc, struct CDnnErrorInfo* errorInfo )
{
	if( dnnDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnDesc parameter.", errorInfo );
		return false;
	}
	return static_cast<const CDnnDescImpl*>( dnnDesc )->RunOnce( errorInfo );
}

const char* GetOutputName( const struct CDnnDesc* dnnDesc, int index, struct CDnnErrorInfo* errorInfo )
{
	if( dnnDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnDesc parameter.", errorInfo );
		return nullptr;
	}

	const CDnnDescImpl* dnn = static_cast<const CDnnDescImpl*>( dnnDesc );
	if( index < 0 || index >= dnn->OutputCount ) {
		initErrorInfo( DET_InvalidParameter, "Invalid index.", errorInfo );
		return nullptr;
	}

	return dnn->GetOutputName( index );
}

const struct CDnnBlobDesc* GetOutputBlob( const struct CDnnDesc* dnnDesc, int index, struct CDnnErrorInfo* errorInfo )
{
	if( dnnDesc == 0 ) {
		initErrorInfo( DET_InvalidParameter, "Invalid CDnnDesc parameter.", errorInfo );
		return nullptr;
	}

	const CDnnDescImpl* descImpl = static_cast<const CDnnDescImpl*>( dnnDesc );

	if( index < 0 || index >= descImpl->OutputCount ) {
		initErrorInfo( DET_InvalidParameter, "Invalid index.", errorInfo );
		return nullptr;
	}

	CPtr<CDnnBlob> blob = descImpl->GetOutputBlob( index );
	if( blob == 0 ) {
		initErrorInfo( DET_RunDnnError, "There is no output blob yet.", errorInfo );
		return nullptr;
	}

	const CDnnMathEngineDescImpl* mathEngineDescImpl = static_cast<const CDnnMathEngineDescImpl*>( descImpl->MathEngine );

	try {
		return FINE_DEBUG_NEW CDnnBlobDescImpl( blob, mathEngineDescImpl );
#ifdef NEOML_USE_FINEOBJ
	} catch( CException* e ) {
		initErrorInfo( DET_InternalError, e->MessageText().CreateString( CP_UTF8 ), errorInfo );
		delete e;
	}
#else
	} catch( std::exception& e ) {
		initErrorInfo( DET_InternalError, e.what(), errorInfo );
	}
#endif

	return nullptr;
}

} // extern "C"
