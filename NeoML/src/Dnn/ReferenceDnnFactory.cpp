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
class CReferenceDnnInfo final {
public:
	CReferenceDnnInfo( CRandom rand, CReferenceDnnFactory& ref ) : random( std::move( rand ) ), owner( ref )
	{ owner.counter += 1; }

	~CReferenceDnnInfo()
	{ owner.counter -= 1; }

	CRandom& Random() { return random; }

private:
	// Stores the dnn's own external random class inside this dnn class
	CRandom random;
	// The factory accounts the number of owned reference dnns
	CReferenceDnnFactory& owner;
};

void CReferenceDnnInfoDeleter::operator()( CReferenceDnnInfo* info ) { if( info != nullptr ) { delete info; } }

//---------------------------------------------------------------------------------------------------------

CReferenceDnnFactory::CReferenceDnnFactory( IMathEngine& mathEngine, CArchive& archive, int seed, bool optimizeDnn ) :
	CReferenceDnnFactory( MakeCPtrOwner<CReferenceDnnInfo>( CRandom( seed ), *this ), mathEngine )
{
	serialize( archive, optimizeDnn );
}

CReferenceDnnFactory::CReferenceDnnFactory( IMathEngine& mathEngine, const CDnn& dnn, bool optimizeDnn ) :
	CReferenceDnnFactory( MakeCPtrOwner<CReferenceDnnInfo>( dnn.Random(), *this ), mathEngine )
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
	CReferenceDnnFactory( MakeCPtrOwner<CReferenceDnnInfo>( dnn.Random(), *this ), dnn.GetMathEngine() )
{
	auto& mathEngineCopy = dnn.GetMathEngine();
	auto& randomCopy = dnn.Random();

	// Allow to createReferenceDnn() this time
	NeoAssert( !dnn.IsReferenceDnn() );
	NeoAssert( originalDnn.IsReferenceDnn() );
	swap( dnn.referenceDnnInfo, originalDnn.referenceDnnInfo );

	// Copy state with moving of the paramBlobs
	dnn.createReferenceDnn( originalDnn, TPtrOwnerReferenceDnnInfo{} );
	if( optimizeDnn == true ) {
		( void ) OptimizeDnn( originalDnn );
	}

	// And revert back the restrictions
	originalDnn.DisableLearning();
	swap( dnn.referenceDnnInfo, originalDnn.referenceDnnInfo );
	NeoAssert( originalDnn.IsReferenceDnn() );
	NeoAssert( !dnn.IsReferenceDnn() );

	dnn.~CDnn(); // Destroy used pointers in the arg dnn
	new( &dnn ) CDnn( randomCopy, mathEngineCopy ); // Ensure, the dtor will be called moramlly
}

CReferenceDnnFactory::CReferenceDnnFactory( CPtrOwner<CReferenceDnnInfo> referenceDnnInfo, IMathEngine& mathEngine ) :
	originalDnn( referenceDnnInfo->Random(), mathEngine )
{
	// Enable this pointer is to add the restriction to change this dnn.
	// Used for the original dnn and all reference dnns
	originalDnn.referenceDnnInfo = referenceDnnInfo.Detach();
	NeoAssertMsg( mathEngine.GetType() == MET_Cpu, "CReferenceDnnFactory: Allowed only for CPU mathEngine" );
}

CReferenceDnnFactory::~CReferenceDnnFactory()
{
	NeoAssertMsg( counter == 0, "CReferenceDnnFactory: Cannot be destroyed before any reference dnns" );
}

void CReferenceDnnFactory::serialize( CArchive& archive, bool optimizeDnn )
{
	// Allow to Serialize() this time
	TPtrOwnerReferenceDnnInfo tmp;
	swap( tmp, originalDnn.referenceDnnInfo );

	NeoAssert( archive.IsLoading() );
	originalDnn.Serialize( archive );
	archive.Close();

	if( optimizeDnn == true ) {
		( void ) OptimizeDnn( originalDnn );
	}

	// And revert back the restrictions
	originalDnn.DisableLearning();
	swap( tmp, originalDnn.referenceDnnInfo );
	NeoAssert( originalDnn.IsReferenceDnn() );
}

CPtrOwner<CDnn> CReferenceDnnFactory::CreateReferenceDnn()
{
	TPtrOwnerReferenceDnnInfo referenceDnnInfo( new CReferenceDnnInfo( originalDnn.Random(), *this ) );
	CPtrOwner<CDnn> referenceDnn( new CDnn( referenceDnnInfo->Random(), originalDnn.GetMathEngine() ) );

	originalDnn.createReferenceDnn( *referenceDnn, std::move( referenceDnnInfo ) );

	NeoAssertMsg( counter > 0, "CReferenceDnnFactory: non-accounted reference dnn" );
	NeoAssertMsg( !referenceDnn->IsLearningEnabled(), "CReferenceDnnFactory: learning enabled for reference dnn" );
	return referenceDnn;
}

} // namespace NeoML
