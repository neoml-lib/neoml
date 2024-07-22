/* Copyright © 2024 ABBYY

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

#include <NeoML/Dnn/Dnn.h>
#ifdef NEOML_COMPACT // No optimizations in mobile assembly
namespace NeoML {
static void OptimizeDnn( CDnn& ) {}
}
#else  // !NEOML_COMPACT
#include <NeoML/Dnn/DnnOptimization.h>
#endif // !NEOML_COMPACT

namespace NeoML {

// Internal technical class
struct CReferenceDnnInfo final {
	CRandom Random; // Stores the dnn's own external random class inside this dnn class
	CPtr<CReferenceDnnFactory> Owner; // The factory accounts the number of owned reference dnns

	CReferenceDnnInfo( CRandom rand, CPtr<CReferenceDnnFactory> ptr ) :
		Random( std::move( rand ) ), Owner( std::move( ptr ) ) {}
};

void CReferenceDnnInfoDeleter::operator()( CReferenceDnnInfo* info ) { delete info; }

//---------------------------------------------------------------------------------------------------------------------

CReferenceDnnFactory::CReferenceDnnFactory( IMathEngine& mathEngine, CArchive& archive, int seed, bool optimizeDnn ) :
	CReferenceDnnFactory( CRandom( seed ), mathEngine )
{
	serialize( archive, optimizeDnn );
}

CReferenceDnnFactory::CReferenceDnnFactory( IMathEngine& mathEngine, const CDnn& dnn, bool optimizeDnn ) :
	CReferenceDnnFactory( dnn.Random(), mathEngine )
{
	// Copy dnn using serialization to get the new dnn of necessary life time
	CMemoryFile file;
	CArchive archive( &file, CArchive::store );
	const_cast<CDnn&>( dnn ).Serialize( archive );
	archive.Close();
	file.SeekToBegin();

	archive.Open( &file, CArchive::load );
	serialize( archive, optimizeDnn );
}

CReferenceDnnFactory::CReferenceDnnFactory( CDnn&& dnn, bool optimizeDnn ) :
	CReferenceDnnFactory( dnn.Random(), dnn.GetMathEngine() )
{
	auto& mathEngineCopy = dnn.GetMathEngine();
	auto& randomCopy = dnn.Random();

	// Allow to createReferenceDnn() from the non-reference dnn this time
	NeoAssert( !dnn.IsReferenceDnn() );
	NeoAssert( Origin->Dnn.IsReferenceDnn() );

	// Temporal convert an ordinary dnn to a reference dnn
	// by capturing non-empty referenceDnnInfo from the original dnn
	swap( dnn.referenceDnnInfo, Origin->Dnn.referenceDnnInfo );
	// The original dnn has empty referenceDnnInfo now

	// Copy state with moving of the paramBlobs
	initializeReferenceDnn( dnn, Origin->Dnn, TPtrOwnerReferenceDnnInfo{} );
	if( optimizeDnn == true ) {
		( void ) OptimizeDnn( Origin->Dnn );
	}
	// The original dnn still has empty referenceDnnInfo

	// Convert everything back
	// - set back the original dnn as a reference dnn with its referenceDnnInfo
	Origin->Dnn.DisableLearning();
	swap( dnn.referenceDnnInfo, Origin->Dnn.referenceDnnInfo );
	// - set the dnn as ordinary and of an empty state
	// Destroy used pointers in the dnn
	dnn.~CDnn();
	// Ensure the dtor will be called normally outside
	new( &dnn ) CDnn( randomCopy, mathEngineCopy );

	// Check the consistence
	NeoAssert( Origin->Dnn.IsReferenceDnn() );
	NeoAssert( !dnn.IsReferenceDnn() );
}

CReferenceDnnFactory::CReferenceDnnFactory( CRandom random, IMathEngine& mathEngine )
{
	NeoAssertMsg( mathEngine.GetType() == MET_Cpu, "CReferenceDnnFactory: Allowed only for CPU mathEngine" );

	TPtrOwnerReferenceDnnInfo originDnnInfo( new CReferenceDnnInfo( std::move( random ), /*factory*/nullptr ) );
	// Factory pointer in the origin dnn should be 0 to avoid cyclic references

	Origin = new CDnnReference( originDnnInfo->Random, mathEngine );
	// Enable this pointer is to add the restriction to change this dnn.
	// Used for the origin dnn and all reference dnns
	Origin->Dnn.referenceDnnInfo = std::move( originDnnInfo );
}

void CReferenceDnnFactory::serialize( CArchive& archive, bool optimizeDnn )
{
	// Allow to Serialize() this time
	TPtrOwnerReferenceDnnInfo tmp;
	swap( tmp, Origin->Dnn.referenceDnnInfo );

	NeoAssert( archive.IsLoading() );
	Origin->Dnn.Serialize( archive );
	archive.Close();

	if( optimizeDnn == true ) {
		( void ) OptimizeDnn( Origin->Dnn );
	}

	// And revert back the restrictions
	Origin->Dnn.DisableLearning();
	swap( tmp, Origin->Dnn.referenceDnnInfo );
	NeoAssert( Origin->Dnn.IsReferenceDnn() );
}

CPtr<CDnnReference> CReferenceDnnFactory::CreateReferenceDnn( bool getOriginDnn )
{
	if( getOriginDnn ) {
		return Origin;
	}
	TPtrOwnerReferenceDnnInfo referenceDnnInfo( new CReferenceDnnInfo( Origin->Dnn.Random(), this ) );
	CPtr<CDnnReference> reference( new CDnnReference( referenceDnnInfo->Random, Origin->Dnn.GetMathEngine() ) );

	initializeReferenceDnn( Origin->Dnn, reference->Dnn, std::move( referenceDnnInfo ) );
	NeoAssert( reference->Dnn.IsReferenceDnn() );

	NeoAssertMsg( !reference->Dnn.IsLearningEnabled(), "CReferenceDnnFactory: learning enabled for reference dnn" );
	return reference;
}

void CReferenceDnnFactory::initializeReferenceDnn( CDnn& dnn, CDnn& newDnn, TPtrOwnerReferenceDnnInfo&& info )
{
	// To initialize the newDnn from the dnn without fair coping the paramBlobs
	// Check the dnn should be a CDnnReference, its referenceDnnInfo should be != 0
	NeoAssert( dnn.IsReferenceDnn() );

	CMemoryFile file;
	for( CPtr<CBaseLayer>& layer : dnn.layers ) {
		file.SeekToBegin();
		{
			CArchive archive( &file, CArchive::store );
			SerializeLayer( archive, dnn.mathEngine, layer ); // if referenceDnnInfo != 0 to do not duplicate paramBlobs
		}
		file.SeekToBegin();
		CPtr<CBaseLayer> copyLayer;
		{
			CArchive archive( &file, CArchive::load );
			SerializeLayer( archive, dnn.mathEngine, copyLayer );
			layer->transferParamsBlob( *copyLayer );
		}
		newDnn.AddLayer( *copyLayer );
	}

	// Convert the newDnn to CDnnReference
	newDnn.referenceDnnInfo = std::move( info );
	newDnn.DisableLearning();
}

} // namespace NeoML
