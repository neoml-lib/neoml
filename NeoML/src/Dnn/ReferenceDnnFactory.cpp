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

namespace NeoML {

CReferenceDnnFactory::CReferenceDnnFactory( IMathEngine& mathEngine, CArchive& archive, int seed ) :
	CReferenceDnnFactory( new CReferenceDnnInfo( CRandom( seed ), nullptr ), mathEngine )
{
	serialize( archive );
}

CReferenceDnnFactory::CReferenceDnnFactory( IMathEngine& mathEngine, const CDnn& dnn ) :
	CReferenceDnnFactory( new CReferenceDnnInfo( const_cast<CDnn&>( dnn ).Random(), nullptr ), mathEngine )
{
	// Copy dnn using serialization to get the new dnn of necessary life time
	CMemoryFile file;
	CArchive archive( &file, CArchive::store );
	const_cast<CDnn&>( dnn ).Serialize( archive );
	archive.Close();
	file.SeekToBegin();

	archive.Open( &file, CArchive::load );
	serialize( archive );
}

CReferenceDnnFactory::CReferenceDnnFactory( CReferenceDnnInfo* referenceDnnInfo, IMathEngine& mathEngine ) :
	originalDnn( referenceDnnInfo->Random(), mathEngine )
{
	// Enable this pointer is to add the restriction to change this dnn.
	// Used for the original dnn and all reference dnns
	originalDnn.referenceDnnInfo = referenceDnnInfo;
	NeoAssertMsg( mathEngine.GetType() == MET_Cpu, "CReferenceDnnFactory: Allowed only for CPU mathEngine" );
}

CReferenceDnnFactory::~CReferenceDnnFactory()
{
	NeoAssertMsg( counter == 0, "CReferenceDnnFactory: Cannot be destroyed before any reference dnns" );
}

void CReferenceDnnFactory::serialize( CArchive& archive )
{
	// Allow to Serialize() this time
	CReferenceDnnInfo* tmp = nullptr;
	swap( tmp, originalDnn.referenceDnnInfo );

	NeoAssert( archive.IsLoading() );
	originalDnn.Serialize( archive );
	archive.Close();

	// And revert back the restrictions
	originalDnn.DisableLearning();
	swap( tmp, originalDnn.referenceDnnInfo );
	NeoAssert( originalDnn.referenceDnnInfo != nullptr );
}

CPtrOwner<CDnn> CReferenceDnnFactory::CreateReferenceDnn()
{
	CDnn* referenceDnn = originalDnn.createReferenceDnn( *this );
	++counter;
	NeoAssertMsg( !referenceDnn->IsLearningEnabled(), "CReferenceDnnFactory: learning enabled for reference dnn" );
	return CPtrOwner<CDnn>( referenceDnn );
}

void CReferenceDnnFactory::destroyReferenceDnn()
{
	NeoAssertMsg( counter > 0, "CReferenceDnnFactory: Cannot be destroyed non reference dnns" );
	--counter;
}

} // namespace NeoML
