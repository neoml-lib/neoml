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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// CSubSequenceLayer selects a subsequence from the input sequence
class NEOML_API CSubSequenceLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CSubSequenceLayer )
public:
	explicit CSubSequenceLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Sets the starting point. If it's < 0, the starting point will be counted from the end of sequence, 
	// with `-1` standing for the last element of the original sequence
	int GetStartPos() const  { return startPos; }
	void SetStartPos(int _startPos);

	// The subsequence length. If it's < 0, the subsequence will be in reverse order
	// If it is too large to fit into the sequence when starting in the specified position, 
	// the subsequence actually extracted will be shorter
	int GetLength() const { return length; }
	void SetLength(int _length);

	void SetReverse() { SetStartPos(-1); SetLength(INT_MIN); }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	int startPos;
	int length;
	// The subsequence objects' indices (used on backward pass)
	CPtr<CDnnBlob> indices;

	void getSequenceInfo(int& sequenceStart, int& subSequenceLength) const;
};

} // namespace NeoML
