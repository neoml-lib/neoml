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

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

//---------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

class CGraphGeneratorTest : public CNeoMLTestFixture {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

class CArc {
public:
	typedef int Quality;
	CArc( int _begin, int _end ) : begin( _begin ), end( _end ) {}

	int InitialCoord() const { return begin; }
	int FinalCoord() const { return end; }
	Quality ArcQuality() const { return 1; }

private:
	int begin;
	int end;
};

class CGraph: public CLdGraph<CArc> {
public:
	typedef CArc GraphArc;
	typedef int Quality;
	
	CGraph() : CLdGraph<CArc>( 0, 6 )
	{
		InsertArc( new CArc( 0, 1 ) );
		InsertArc( new CArc( 0, 2 ) );
		InsertArc( new CArc( 0, 3 ) );
		InsertArc( new CArc( 1, 3 ) );
		InsertArc( new CArc( 1, 5 ) );
		InsertArc( new CArc( 2, 3 ) );
		InsertArc( new CArc( 2, 6 ) );
		InsertArc( new CArc( 3, 4 ) );
		InsertArc( new CArc( 4, 6 ) );
		InsertArc( new CArc( 5, 6 ) );
	}
};

class CEdge {
public:
	typedef double Quality;

	CEdge() : cost( 10000 ) {}
	CEdge( int, int, double _cost ) : cost( _cost ) {}
	double Penalty() const { return cost; }

private:
	double cost;
};

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

TEST( CGraphGeneratorTest, GraphGeneratorCommonTest )
{
	CGraph g;
	g.CalculateBestPathQuality(INT_MIN / 2);

	CGraphGenerator<CGraph> generator( &g, 0, -10000 );

	CArray<const CArc*> path;

	EXPECT_EQ( generator.NextPathQuality(), 4 );

	generator.GetNextPath( path );

	EXPECT_EQ( generator.NextPathQuality(), 4 );

	generator.GetNextPath( path );
	generator.GetNextPath( path );
	generator.GetNextPath( path );
	
	EXPECT_EQ( generator.GetNextPath( path ), true );
	
	EXPECT_EQ( path.Size(), 2 ); // last path

	EXPECT_EQ( generator.GetNextPath( path ), false );
}

TEST( CGraphGeneratorTest, MatchingGeneratorCommonTest )
{
	const int leftSize = 5;
	const int rightSize = 5;

	CMatchingGenerator<CEdge> generator( leftSize, rightSize, 0, 1000000000 );
	CVariableMatrix<CEdge>& matrix = generator.PairMatrix();

	for( int i = 0; i < leftSize; i++ ) {
		for( int j = 0; j < rightSize; j++ ) {
			matrix( i, j ) = CEdge( i, j, 10000 );
		}
	}

	matrix( 0, 0 ) = CEdge( 0, 0, 1 );
	matrix( 0, 1 ) = CEdge( 0, 1, 1 );
	matrix( 0, 2 ) = CEdge( 0, 2, 1 );

	matrix( 1, 1 ) = CEdge( 1, 1, 1 );
	matrix( 1, 2 ) = CEdge( 1, 2, 1 );

	matrix( 2, 0 ) = CEdge( 2, 0, 1 );
	matrix( 2, 3 ) = CEdge( 2, 3, 1 );

	matrix( 3, 3 ) = CEdge( 3, 3, 1 );
	matrix( 3, 4 ) = CEdge( 3, 4, 1 );

	matrix( 4, 4 ) = CEdge( 4, 4, 1 );

	generator.Build();

	CArray<CEdge> res;
	double expectedQuality = generator.EstimateNextMatchingPenalty();
	EXPECT_EQ( Round( expectedQuality ), 5 );
	double resQuality = generator.GetNextMatching( res );
	EXPECT_EQ( Round( resQuality ), 5 );
}

TEST( CGraphGeneratorTest, MatchingGeneratorUniqueMatches6x8 )
{
    struct CMyPair {
        typedef int Quality;

        int LeftIndex = -1;
        int RightIndex = -1;
        Quality Score = 0;

        Quality Penalty() const { return 1 - Score; }
    };
    const int numLeft = 6;
    const int numRight = 8;
    const int pairScores[numRight][numLeft] = {
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1}
    };
	CMatchingGenerator<CMyPair> generator( numLeft, numRight, 0, INT_MAX );
    for( int leftInd = 0; leftInd < numLeft; ++leftInd ) {
        for( int rightInd = 0; rightInd < numRight; ++rightInd ) {
            CMyPair& pair = generator.PairMatrix()(leftInd, rightInd);
            pair.LeftIndex = leftInd;
            pair.RightIndex = rightInd;
            pair.Score = pairScores[rightInd][leftInd];
        }
    }
    generator.Build();
    CArray<CMyPair> matching;
    generator.GetNextMatching( matching );
    const auto& FindMatchForLeft = [&matching]( int leftInd )
    {
        for( const CMyPair& pair : matching ) {
            if( pair.LeftIndex == leftInd ) {
                return pair.RightIndex;
            }
        }
        return NotFound;
    };
    EXPECT_EQ( 5, FindMatchForLeft( 3 ) );
    EXPECT_EQ( 6, FindMatchForLeft( 4 ) );
    EXPECT_EQ( 7, FindMatchForLeft( 5 ) );
}
