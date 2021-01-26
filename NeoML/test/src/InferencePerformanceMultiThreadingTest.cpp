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

#include <TestFixture.h>

#include <thread>
#include <future>

namespace NeoMLTest {

enum class TTimeType {
	TT_Undefined = 0,
	TT_Average,
	TT_Total
};

struct CDnnInferencePerformanceTestParam {
	int RunCount;
	int ThreadCount;
	TTimeType TimeType;
	CString Name;
	CArray<CString> Sources;
	CArray<CString> Sinks;

	CDnnInferencePerformanceTestParam( const CDnnInferencePerformanceTestParam& other ) :
		RunCount( other.RunCount ),
		ThreadCount( other.ThreadCount ),
		TimeType( other.TimeType ),
		Name( other.Name )
	{
		other.Sources.CopyTo( Sources );
		other.Sinks.CopyTo( Sinks );
	}

	CDnnInferencePerformanceTestParam( int runCount, int threadCount, TTimeType timeType, const CString& name, std::initializer_list<CString> sources, std::initializer_list<CString> sinks ) :
		RunCount( runCount ),
		ThreadCount( threadCount ),
		TimeType( timeType ),
		Name( name )
	{
		Sources.SetBufferSize( to<int>( sources.size() ) );
		Sinks.SetBufferSize( to<int>( sinks.size() ) );

		for( int i = 0; i < to<int>(sources.size()); i++ ) {
			Sources.Add( sources.begin()[i] );
		}

		for( int i = 0; i < to<int>(sinks.size()); i++ ) {
			Sinks.Add( sinks.begin()[i] );
		}
	}
};

::std::ostream& operator<<( ::std::ostream& os, const CDnnInferencePerformanceTestParam& params )
{
	return os << params.Name;
}

using ResultType = std::unique_ptr<IPerformanceCounters>;

class CDnnInferencePerformanceTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CDnnInferencePerformanceTestParam> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture();

	static void LoadCnn( const CDnnInferencePerformanceTestParam& param, CDnn& cnn );
	static CPtr<CDnnBlob> LoadBlob( const CString& fileName, IMathEngine& mathEngine );

	static ResultType Run( const CDnnInferencePerformanceTestParam& param, IMathEngine& mathEngine );

	static CRandom& GetRandom() { return random; }

private:
	static CRandom random;
};

CRandom CDnnInferencePerformanceTest::random;

void CDnnInferencePerformanceTest::DeinitTestFixture()
{
}

void CDnnInferencePerformanceTest::LoadCnn( const CDnnInferencePerformanceTestParam& param, CDnn& cnn )
{
	CArchiveFile file( GetTestDataFilePath( "data", param.Name + ".cnnarch" ), CArchive::load, GetPlatformEnv() );
	CArchive archive( &file, CArchive::SD_Loading );
	archive.Serialize( cnn );
	archive.Close();
	file.Close();

	for( int i = 0; i < param.Sources.Size(); i++ ) {
		CPtr<CSourceLayer> sourceLayer = CheckCast<CSourceLayer>( cnn.GetLayer( param.Sources[i] ) );
		sourceLayer->SetBlob( LoadBlob( param.Name + "." + param.Sources[i] + ".input", cnn.GetMathEngine() ) );
	}
}

CPtr<CDnnBlob> CDnnInferencePerformanceTest::LoadBlob( const CString& fileName, IMathEngine& mathEngine )
{
	CArchiveFile file( GetTestDataFilePath( "data", fileName ), CArchive::load, GetPlatformEnv() );
	CArchive archive( &file, CArchive::SD_Loading );
	CPtr<CDnnBlob> blob = new CDnnBlob( mathEngine );
	blob->Serialize( archive );
	return blob;
}

template <typename T>
static void checkResults( const CPtr<CDnnBlob>& actualBlob, const CPtr<CDnnBlob>& expectedBlob )
{
	CArray<T> expectedData;
	CArray<T> actualData;
	expectedData.SetSize( expectedBlob->GetDataSize() );
	actualData.SetSize( actualBlob->GetDataSize() );
	expectedBlob->CopyTo( expectedData.GetPtr() );
	actualBlob->CopyTo( actualData.GetPtr() );

	ASSERT_EQ( expectedData.Size(), actualData.Size() );
	for( int i = 0; i < expectedData.Size(); i++ ) {
		ASSERT_NEAR( expectedData[i], actualData[i], 1E-2 ) << i;
	}
}

ResultType CDnnInferencePerformanceTest::Run( 
	const CDnnInferencePerformanceTestParam& param, IMathEngine& mathEngine )
{
	CDnn cnn( GetRandom(), mathEngine );
	
	LoadCnn( param, cnn );

	cnn.RunOnce();

	CObjectArray<CSinkLayer> sinkLayers;
	CArray<CArray<float>> sinkBlobData;
	sinkBlobData.SetSize( param.Sinks.Size() );
	for( int i = 0; i < param.Sinks.Size(); i++ ) {
		sinkLayers.Add( CheckCast<CSinkLayer>( cnn.GetLayer( param.Sinks[i] ) ) );
		CPtr<CDnnBlob> sinkBlob = sinkLayers.Last()->GetBlob();
		sinkBlobData[i].SetSize( sinkBlob->GetDataSize() );
	}

	std::unique_ptr<IPerformanceCounters> counters( mathEngine.CreatePerformanceCounters() );
	counters->Synchronise();

	for( int run = 0; run < param.RunCount; ++run ) {
		cnn.RunOnce();

		for( int i = 0; i < param.Sinks.Size(); i++ ) {
			CPtr<CDnnBlob> sinkBlob = sinkLayers[i]->GetBlob();
			sinkBlob->CopyTo( sinkBlobData[i].GetPtr() );
		}
	}
	counters->Synchronise();
	
	// Check results
	for( int i = 0; i < param.Sinks.Size(); i++ ) {
		CPtr<CDnnBlob> expectedBlob = LoadBlob( param.Name + "." + param.Sinks[i] + ".output", cnn.GetMathEngine() );
		CPtr<CDnnBlob> actualBlob = sinkLayers[i]->GetBlob();

		if( expectedBlob->GetDataType() == CT_Float ) {
			checkResults<float>( actualBlob, expectedBlob );
		} else if( expectedBlob->GetDataType() == CT_Int ) {
			checkResults<int>( actualBlob, expectedBlob );
		} else {
			EXPECT_TRUE( false );
		}
	}

	mathEngine.CleanUp();

	return counters;
}

} // namespace NeoMLTest

using namespace NeoML;
using namespace NeoMLTest;

//------------------------------------------------------------------------------------------------------------

TEST_P( CDnnInferencePerformanceTest, OneMathEngine )
{
	const auto& param = GetParam();

	auto& mathEngine = MathEngine();

	std::vector<std::future<ResultType>> results;
	results.reserve( param.ThreadCount );

	for( int i = 0; i < param.ThreadCount; ++i ) {
		results.push_back( std::async( std::launch::async, Run, std::ref( param ), std::ref( mathEngine ) ) );
	}	
	
	try {
		for( auto& result : results ) {
			bool useavg = ( param.TimeType == TTimeType::TT_Average );
			auto counters = result.get();
			for( const auto& counter : *counters ) {
				GTEST_LOG_( INFO ) << param.Name << " " << counter.Name << ": " << 
					( useavg ? counter.Value / param.RunCount : counter.Value );
			}
		}
	} catch( std::exception& e ) {
		GTEST_LOG_( ERROR ) << e.what();
		throw;
	}
}

TEST_P( CDnnInferencePerformanceTest, LocalMathEngine )
{
	const auto& param = GetParam();

	std::vector<std::future<ResultType>> results;
	results.reserve( param.ThreadCount );

	std::vector<std::unique_ptr<IMathEngine>> mathEngines;
	mathEngines.reserve( param.ThreadCount );

	constexpr std::size_t memoryLimit = 256 * 1024 * 1024;

	DeleteMathEngine();

	for( int i = 0; i < param.ThreadCount; ++i ) {
		auto mathEngine = CreateMathEngine( MathEngineType(), memoryLimit, 1 );
		ASSERT_TRUE( mathEngine != nullptr ) << i;
		mathEngines.emplace_back( mathEngine );
		results.push_back( std::async( std::launch::async, Run, std::ref( param ), std::ref( *mathEngine ) ) );
	}

	try {
		for( auto& result : results ) {
			bool useavg = (param.TimeType == TTimeType::TT_Average);
			auto counters = result.get();
			for( const auto& counter : *counters ) {
				GTEST_LOG_(INFO) << param.Name << " " << counter.Name << ": " <<
					(useavg ? counter.Value / param.RunCount : counter.Value);
			}
		}
	} catch( std::exception& e ) {
		GTEST_LOG_( ERROR ) << e.what();
		throw;
	}
}

INSTANTIATE_TEST_CASE_P( CDnnInferencePerformanceTestInstantiation, CDnnInferencePerformanceTest,
	::testing::Values(
		CDnnInferencePerformanceTestParam(
			1000, 
			4, 
			TTimeType::TT_Total,
			"MobileNetV2Cifar10", // 32x32x3
			{ "in" },
			{ "out" }
		)
	)
);
