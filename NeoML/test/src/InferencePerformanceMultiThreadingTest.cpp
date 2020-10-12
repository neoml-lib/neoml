#include <common.h>
#pragma hdrstop

#include <TestFixture.h>

#if FINE_PLATFORM( FINE_IOS )
#include <sys/kdebug_signpost.h>
#endif

#include <thread>
#include <future>

namespace NeoMLTest {
    
enum TTimeType {
	TT_Undefined = 0,
	TT_Average,
	TT_Total
};

// Параметры теста.
struct CDnnInferencePerformanceTestParam {
	int RunCount;
	TTimeType TimeType;
	CString Name;
	CArray<CString> Sources;
	CArray<CString> Sinks;

	CDnnInferencePerformanceTestParam( const CDnnInferencePerformanceTestParam& other ) :
		RunCount( other.RunCount ),
		TimeType( other.TimeType ),
		Name( other.Name )
	{
		other.Sources.CopyTo( Sources );
		other.Sinks.CopyTo( Sinks );
	}

	CDnnInferencePerformanceTestParam( int runCount, TTimeType timeType, const CString& name, std::initializer_list<CString> sources, std::initializer_list<CString> sinks ) :
		RunCount( runCount ),
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

class CDnnInferencePerformanceTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<CDnnInferencePerformanceTestParam> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture();

	static void LoadCnn( const CDnnInferencePerformanceTestParam& param, CDnn& cnn );
	static CPtr<CDnnBlob> LoadBlob( const CString& fileName );

	static void TestRunOnce( const CDnnInferencePerformanceTestParam& param );

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
		sourceLayer->SetBlob( LoadBlob( param.Name + "." + param.Sources[i] + ".input" ) );
	}
}

CPtr<CDnnBlob> CDnnInferencePerformanceTest::LoadBlob( const CString& fileName )
{
	CArchiveFile file( GetTestDataFilePath( "data", fileName ), CArchive::load, GetPlatformEnv() );
	CArchive archive( &file, CArchive::SD_Loading );
	CPtr<CDnnBlob> blob = new CDnnBlob( MathEngine() );
	blob->Serialize( archive );
	return blob;
}

void CDnnInferencePerformanceTest::TestRunOnce( const CDnnInferencePerformanceTestParam& param )
{
	CDnn cnn( GetRandom(), MathEngine() );
	
	LoadCnn( param, cnn );

	// Первый проход не показательный.
	cnn.RunOnce();

	CObjectArray<CSinkLayer> sinkLayers;
	CArray<CArray<float>> sinkBlobData;
	sinkBlobData.SetSize( param.Sinks.Size() );
	for( int i = 0; i < param.Sinks.Size(); i++ ) {
		sinkLayers.Add( CheckCast<CSinkLayer>( cnn.GetLayer( param.Sinks[i] ) ) );
		CPtr<CDnnBlob> sinkBlob = sinkLayers.Last()->GetBlob();
		sinkBlobData[i].SetSize( sinkBlob->GetDataSize() );
	}

	// Замеры скорости:
	auto counters = cnn.GetMathEngine().CreatePerformanceCounters();
	counters->Synchronise();

	for( int run = 0; run < param.RunCount; ++run ) {
		cnn.RunOnce();

		// вызовем sync (чтобы правильно измерить время)
		for( int i = 0; i < param.Sinks.Size(); i++ ) {
			CPtr<CDnnBlob> sinkBlob = sinkLayers[i]->GetBlob();
			sinkBlob->CopyTo( sinkBlobData[i].GetPtr() );
		}
	}
	counters->Synchronise();
	bool useavg = (param.TimeType == TT_Average);
	GTEST_LOG_(INFO) << param.Name << " peak memory usage: " << ( MathEngine().GetPeakMemoryUsage() / 1024) << " Kb";
	for( const auto& counter : *counters ) {
		GTEST_LOG_(INFO) << param.Name << " " << counter.Name << ": " << (useavg ? counter.Value / param.RunCount : counter.Value);
	}
	delete counters;
	
	// Проверка правильности результата:
	CArray<float> expectedDataFloat;
	CArray<int> expectedDataInt;
	CArray<float> actualDataFloat;
	CArray<int> actualDataInt;
	for( int i = 0; i < param.Sinks.Size(); i++ ) {
		CPtr<CDnnBlob> expectedBlob = LoadBlob( param.Name + "." + param.Sinks[i] + ".output" );
		CPtr<CDnnBlob> actualBlob = sinkLayers[i]->GetBlob();

		if( expectedBlob->GetDataType() == CT_Float ) {
			expectedDataFloat.SetSize( expectedBlob->GetDataSize() );
			actualDataFloat.SetSize( actualBlob->GetDataSize() );
			expectedBlob->CopyTo( expectedDataFloat.GetPtr() );
			actualBlob->CopyTo( actualDataFloat.GetPtr() );
			CArray<float>& actualData = actualDataFloat;

			ASSERT_EQ( expectedDataFloat.Size(), actualData.Size() );
			for( int j = 0; j < expectedDataFloat.Size(); j++ ) {
				ASSERT_NEAR( expectedDataFloat[j], actualData[j], 1E-2 ) << j;
			}
		} else if( expectedBlob->GetDataType() == CT_Int ) {
			expectedDataInt.SetSize( expectedBlob->GetDataSize() );
			actualDataInt.SetSize( actualBlob->GetDataSize() );
			expectedBlob->CopyTo( expectedDataInt.GetPtr() );
			actualBlob->CopyTo( actualDataInt.GetPtr() );
			CArray<int>& actualData = actualDataInt;

			ASSERT_EQ( expectedDataInt.Size(), actualData.Size() );
			for( int j = 0; j < expectedDataInt.Size(); j++ ) {
				ASSERT_NEAR( expectedDataInt[j], actualData[j], 1E-2 ) << j;
			}
		} else {
			ASSERT_TRUE( false );
		}
	}
}

} // namespace NeoMLTest

using namespace NeoML;
using namespace NeoMLTest;

//------------------------------------------------------------------------------------------------------------

TEST_P(CDnnInferencePerformanceTest, Test)
{
	const auto& param = GetParam();
	
	const int threadCount = 2;

	std::vector<std::future<decltype( TestRunOnce( param ) )>> results;
	results.reserve( threadCount );

	for( int i = 0; i < threadCount; ++i ) {
		results.push_back( std::async( std::launch::async, TestRunOnce, std::ref( param ) ) );
	}	
	
	try {
		for( auto& result : results ) {
			result.get();
		}
	} catch( std::exception& e ) {
		GTEST_LOG_( ERROR ) << e.what();
		throw;
	}

	MathEngine().CleanUp();
}

INSTANTIATE_TEST_CASE_P( CDnnInferencePerformanceTestInstantiation, CDnnInferencePerformanceTest,
	::testing::Values(
		CDnnInferencePerformanceTestParam(1000, TT_Total, "MobileNetV2Cifar10", // 32x32x3
			{ "in" },
			{ "out" }
		)
	)
);
