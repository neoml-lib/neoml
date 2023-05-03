/* Copyright © 2017-2023 ABBYY

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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/SubwordEncoder.h>

namespace NeoML {

// Converts list of subtoken ids to the original text
class NEOML_API CSubwordDecoder {
public:
	CSubwordDecoder( ISubwordEncoder::CParams params, CMap<int, CString>&& idToToken );

	// Convert 'tokenIds' to a list of words
	void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const;

private:
	ISubwordEncoder::CParams params;
	CMap<int, CString> idToToken;

	void removeSpecialTokens( CString& token, bool& hasEow, bool& hasSow ) const;
	bool replaceEowToken( CString& token, const CString& eowToken, const CString& replacement ) const;
	bool replaceSowToken( CString& token, const CString& sowToken, const CString& replacement ) const;
};
} // namespace NeoML
