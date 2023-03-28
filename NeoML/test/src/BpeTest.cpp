/* Copyright © 2022 ABBYY Production LLC

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

static void splitString( const CString& text, CArray<CString>& out, char delimiter = ' ' )
{
	out.DeleteAll();
	bool newStr = true;
	for( int i = 0; i < text.Length(); ++i ) {
		if( newStr ) {
			out.Append();
			newStr = false;
		}
		if( text[i] == delimiter ) {
			newStr = true;
		} else {
			out.Last() += text[i];
		}
	}
}

static CWordDictionary fillDictionary( const CString& text )
{
	CWordDictionary result;

	CArray<CString> split;
	splitString( text, split );
	for( int i = 0; i < split.Size(); ++i ) {
		result.AddWord( split[i] );
	}

	return result;
}

class CBpeTest : public CNeoMLTestFixture {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}

	const int UnkIndex = 0;
};

// --------------------------------------------------------------------------------------------------------------------
// test implementation

TEST_F( CBpeTest, DictionaryTest )
{
	CWordDictionary dictionary;
	dictionary.AddWord( "Lorem", 1 );
	dictionary.AddWord( "ipsum", 3 );
	dictionary.AddWord( "dolor", 1 );
	dictionary.AddWord( "sit", 3 );
	dictionary.AddWord( "amet", 2 );
	dictionary.AddWord( "dolor", 1 );
	dictionary.Finalize( 2 );

	EXPECT_FALSE( dictionary.HasWord( "Lorem" ) );
	EXPECT_TRUE( dictionary.HasWord( "ipsum" ) );
	EXPECT_TRUE( dictionary.HasWord( "dolor" ) );
	EXPECT_TRUE( dictionary.HasWord( "sit" ) );
	EXPECT_TRUE( dictionary.HasWord( "amet" ) );

	ASSERT_EQ( 4, dictionary.Size() );
}

// Check on trivial sample
TEST_F( CBpeTest, TrivialOneWord )
{
	auto dictionary = fillDictionary( "OnlyOneWord" );
	CBytePairEncoderTrainer trainer( 100500, CBytePairEncoderTrainer::TBorderHandling::None );
	auto tokenizer = trainer.Train( dictionary );

	CString correctText = "OnlyOneWord";
	CArray<int> tokenIds, tokenLengths;
	tokenizer->Encode( correctText, tokenIds, tokenLengths );
	ASSERT_EQ( 1, tokenIds.Size() );
	EXPECT_EQ( correctText.Length(), tokenLengths[0] );

	CArray<CString> decoded;
	tokenizer->Decode( tokenIds, decoded );
	ASSERT_EQ( 1, decoded.Size() );
	EXPECT_EQ( correctText, decoded[0] );
}

TEST_F( CBpeTest, TrivialUnknown )
{
	auto dictionary = fillDictionary( "OnlyOneWord" );
	CBytePairEncoderTrainer trainer( 100500, CBytePairEncoderTrainer::TBorderHandling::None );
	auto tokenizer = trainer.Train( dictionary );

	CString unknownText = "UNKNNSYMBLS";
	CArray<int> tokenIds, tokenLengths;
	tokenizer->Encode( unknownText, tokenIds, tokenLengths );
	ASSERT_EQ( unknownText.Length(), tokenIds.Size() );
	for( int i = 0; i < tokenIds.Size(); ++i ) {
		EXPECT_EQ( UnkIndex, tokenIds[i] );
	}
	tokenIds.DeleteAll();
	tokenLengths.DeleteAll();

	CString mixedText = "UNKOnlySYMor";
	tokenizer->Encode( mixedText, tokenIds, tokenLengths );
	// <UNK> <UNK> <UNK> Only <UNK> <UNK> <UNK> o r
	ASSERT_EQ( 3 + 1 + 3 + 2, tokenIds.Size() );
}

TEST_F( CBpeTest, OneLetterBpe )
{
	CString trainText ="qwertyuiopasdfghjklzxcvbnm.";
	auto dictionary = fillDictionary( trainText );
	CBytePairEncoderTrainer trainer( 28, CBytePairEncoderTrainer::TBorderHandling::None );
	auto tokenizer = trainer.Train( dictionary );

	CString testText = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
	" incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex"
	" ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur"
	" sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum .";

	CArray<CString> words;
	splitString( testText, words );

	for( int i = 0; i < words.Size(); ++i ) {
		CArray<int> tokenIds, tokenLengths;
		tokenizer->Encode( words[i], tokenIds, tokenLengths );

		// All words are split into known one-letter tokens
		EXPECT_EQ( words[i].Length(), tokenIds.Size() );
		for( int j = 0; j < tokenIds.Size(); ++j ) {
			EXPECT_NE( UnkIndex, tokenIds[j] );
		}

		CArray<CString> decoded;
		tokenizer->Decode( tokenIds, decoded );
		EXPECT_EQ( words[i], decoded[0] );
	}
}

TEST_F( CBpeTest, DecodeSequence )
{
	CString trainText = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"
		" incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex"
		" ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur"
		" sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum .";
	auto dictionary = fillDictionary( trainText );

	CBytePairEncoderTrainer trainerBow( 50, CBytePairEncoderTrainer::TBorderHandling::BeginOfWord );
	CBytePairEncoderTrainer trainerEow( 50, CBytePairEncoderTrainer::TBorderHandling::EndOfWord );
	CArray<CPtr<IBytePairEncoder>> tokenizers = { trainerBow.Train( dictionary ), trainerEow.Train( dictionary ) };

	CString testText = "mattis pellentesque id nibh tortor id aliquet . tincidunt ornare massa eget egestas purus ."
		" orci phasellus egestas tellus rutrum tellus pellentesque eu tincidunt tortor . et malesuada fames ac turpis ."
		" et netus et malesuada fames . quis ipsum suspendisse ultrices gravida dictum . dictumst quisque sagittis purus sit ."
		" turpis tincidunt id aliquet risus feugiat in ante metus dictum .";

	CArray<CString> words;
	splitString( testText, words );

	for( int k = 0; k < tokenizers.Size(); ++k ) {
		auto&& tokenizer = tokenizers[k];

		CArray<int> wholeSequence;
		for( int i = 0; i < words.Size(); ++i ) {
			CArray<int> tokenIds, tokenLengths;
			tokenizer->Encode( words[i], tokenIds, tokenLengths );
			wholeSequence.Add( tokenIds );
		}

		CArray<CString> decodedSequence;
		tokenizer->Decode( wholeSequence, decodedSequence );
		ASSERT_EQ( words, decodedSequence );
	}
}

TEST_F( CBpeTest, Ambiguous )
{
	IBytePairEncoder::CBPEDictionary dictionary = { "aa", "bb", "ab", "a", "b" };
	CPtr<IBytePairEncoder> tokenizer = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );
	IBytePairEncoder::CParams params;
	tokenizer->Initialize( dictionary, params );

	CArray<int> tokenIds, tokenLengths;
	tokenizer->Encode( "aaa", tokenIds, tokenLengths );
	ASSERT_EQ( 2, tokenLengths.Size() );
	EXPECT_EQ( 2, tokenLengths[0] );
	EXPECT_EQ( 1, tokenLengths[1] );
	tokenIds.DeleteAll();
	tokenLengths.DeleteAll();

	tokenizer->Encode( "aabb", tokenIds, tokenLengths );
	ASSERT_EQ( 2, tokenLengths.Size() );
	EXPECT_EQ( 2, tokenLengths[0] );
	EXPECT_EQ( 2, tokenLengths[1] );
	tokenIds.DeleteAll();
	tokenLengths.DeleteAll();

	tokenizer->Encode( "aaaba", tokenIds, tokenLengths );
	ASSERT_EQ( 3, tokenLengths.Size() );
	EXPECT_EQ( 2, tokenLengths[0] );
	EXPECT_EQ( 2, tokenLengths[1] );
	EXPECT_EQ( 1, tokenLengths[2] );
	tokenIds.DeleteAll();
	tokenLengths.DeleteAll();

	tokenizer->Encode( "aaabbb", tokenIds, tokenLengths );
	ASSERT_EQ( 4, tokenLengths.Size() );
	EXPECT_EQ( 2, tokenLengths[0] );
	EXPECT_EQ( 1, tokenLengths[1] );
	EXPECT_EQ( 2, tokenLengths[2] );
	EXPECT_EQ( 1, tokenLengths[3] );
}

#ifdef NEOML_USE_FINEOBJ
#define BPE_TEST_ASSERT( expr ) \
	try { \
		( expr ); \
		FAIL() << "No exception has been thrown during '" << #expr << "'"; \
	} catch( CInternalError* err ) { \
		err->Delete(); \
	} catch( ... ) { \
		FAIL() << "Wrong exception has been thrown during '" << #expr << "'"; \
	}
#else
#define BPE_TEST_ASSERT( expr ) EXPECT_THROW( ( expr ), CInternalError )
#endif

TEST_F( CBpeTest, LoadIncorrectDictionary )
{
	CPtr<IBytePairEncoder> tokenizer = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );

	IBytePairEncoder::CParams params;
	params.EndOfWordToken = "@"; 
	IBytePairEncoder::CBPEDictionary badDictionary = { "a@a", "a" };
	BPE_TEST_ASSERT( tokenizer->Initialize( badDictionary, params ) );

	// aa@@ is inseparable
	tokenizer = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );
	IBytePairEncoder::CBPEDictionary dictionary = { "aa@@", "a" };
	params.EndOfWordToken = ""; 
	BPE_TEST_ASSERT( tokenizer->Initialize( dictionary, params ) );

	// no single '@@'
	tokenizer = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );
	params.EndOfWordToken = "@@"; 
	dictionary.InsertAt( "aa", 0 );
	BPE_TEST_ASSERT( tokenizer->Initialize( dictionary, params ) );
	dictionary.Add( "@@" );

	// wrong symbol
	tokenizer = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );
	params.StartOfWordToken = "";
	params.EndOfWordToken = "!";
	BPE_TEST_ASSERT( tokenizer->Initialize( dictionary, params ) );
}

TEST_F( CBpeTest, SaveLoadDictionary )
{
	CPtr<IBytePairEncoder> tokenizer = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );

	IBytePairEncoder::CBPEDictionary dictionary = { "aa@", "aa", "a", "@" };
	IBytePairEncoder::CParams params;
	params.EndOfWordToken = "@"; 
	tokenizer->Initialize( dictionary, params );

	CArray<int> tokenIds, tokenLengths;
	tokenizer->Encode( "a", tokenIds, tokenLengths );
	ASSERT_EQ( 2, tokenLengths.Size() );
	EXPECT_EQ( 1, tokenLengths[0] );
	EXPECT_EQ( 0, tokenLengths[1] );
	tokenIds.DeleteAll();
	tokenLengths.DeleteAll();

	tokenizer->Encode( "aa", tokenIds, tokenLengths );
	ASSERT_EQ( 1, tokenLengths.Size() );
	tokenIds.DeleteAll();
	tokenLengths.DeleteAll();	

	CMap<CString, int> outDictionary;
	tokenizer->GetTokenToIdMapping( outDictionary );
	EXPECT_EQ( 5, outDictionary.Size() );
	EXPECT_TRUE( outDictionary.Has( "aa@" ) );
	EXPECT_TRUE( outDictionary.Has( "aa" ) );
	EXPECT_TRUE( outDictionary.Has( "a" ) );
	EXPECT_TRUE( outDictionary.Has( "@" ) );
	EXPECT_TRUE( outDictionary.Has( "<UNK>" ) );
}

TEST_F( CBpeTest, RawBytes )
{
	char allBytes[256];
	for( int i = 0; i < 256; ++i ) {
		allBytes[i] = (char)(i + 1);
	}
	// btw, 0-byte is terminal, so we don't expect to see it in the dictionary

	CString superWord( allBytes );
	CWordDictionary trainingDictionary;
	trainingDictionary.AddWord( superWord, 1 );

	CBytePairEncoderTrainer trainer( 100500, CBytePairEncoderTrainer::TBorderHandling::None, CBytePairEncoderTrainer::TVocabPruning::ByteBPE );
	auto encoder = trainer.Train( trainingDictionary );

	// unk, single-bytes, all prefixes of 'allBytes' (incl. the string itself, excl. the first prefix since it is already counted)
	EXPECT_EQ( 1 + 255 + 254, encoder->Size() );

	CArray<CString> words = { "just word", "кириллица", "♥⅀" };
	CArray<int> encoded, notUsed;
	for( int i = 0; i < words.Size(); ++i ) {
		encoded.DeleteAll();
		encoder->Encode( words[i], encoded, notUsed );
		EXPECT_EQ( NotFound, encoded.Find( encoder->UnknownTokenId() ) );
	}
}

TEST_F( CBpeTest, UnknownId )
{
	CPtr<IBytePairEncoder> tokenizer = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );

	IBytePairEncoder::CBPEDictionary dictionary = { "aa@", "aa", "a", "@" };
	IBytePairEncoder::CParams params;
	params.EndOfWordToken = "@"; 
	tokenizer->Initialize( dictionary, params );

	CArray<int> tokenIds, tokenLengths;
	tokenizer->Encode( "baaa", tokenIds, tokenLengths );
	EXPECT_EQ( tokenizer->UnknownTokenId(), tokenIds.First() );

	const int offset = 5;
	tokenizer = CheckCast<IBytePairEncoder>( CreateModel( BytePairEncoderModelName ) );
	params.UnknownTokenId = offset;
	tokenizer->Initialize( dictionary, params );
	ASSERT_EQ( offset, tokenizer->UnknownTokenId() );

	CArray<int> newTokenIds, newTokenLengths;
	tokenizer->Encode( "baaa", newTokenIds, newTokenLengths );
	ASSERT_EQ( tokenIds.Size(), newTokenIds.Size() );
	EXPECT_EQ( tokenLengths, newTokenLengths );
	for( int i = 0; i < tokenIds.Size(); ++i ) {
		EXPECT_EQ( tokenIds[i] + offset, newTokenIds[i] );
	}
}
