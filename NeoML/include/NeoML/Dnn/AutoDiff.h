/* Copyright © 2017-2021 ABBYY Production LLC

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
#include <NeoML/Dnn/DnnBlob.h>

namespace NeoML {

class CGradientTapeImpl;
class IGradientTape;

class NEOML_API CGradientTape {
public:
	CGradientTape();
	~CGradientTape();

	// Creates tape variable.
	CPtr<const CDnnBlob> Variable( const CDnnBlob& blob );
	CPtr<const CDnnBlob> Variable( const CFloatHandle& data, int height, int width, int depth, int channels );

	// Computes the gradient using operations recorded in this tape.
	CPtr<const CDnnBlob> Gradient( const CDnnBlob& expression, const CDnnBlob& var );

private:
	CPtr<CGradientTapeImpl> impl;

	CGradientTape( const CGradientTape& ) = delete;
	CGradientTape& operator=( const CGradientTape& ) = delete;
};

//------------------------------------------------------------------------------------------------------------

class NEOML_API CTapeBlob : public CDnnBlob {
public:
	CTapeBlob( IGradientTape* tape, const CDnnBlob& blob );
	CTapeBlob( IGradientTape* tape, IMathEngine& mathEngine, const CBlobDesc& desc );
	CTapeBlob( IGradientTape* tape, const CFloatHandle& data, int height, int width, int depth, int channels );

	CPtr<IGradientTape> Tape() const { return tape; }
	
protected:
	virtual ~CTapeBlob();

	// Detach blob from tape.
	void Detach() const;

private:
	mutable CPtr<IGradientTape> tape;

	friend CGradientTapeImpl;
};

//------------------------------------------------------------------------------------------------------------

class ITapeOperation : public IObject {
public:
	virtual CPtr<CDnnBlob> Gradient( const CTapeBlob* var ) const = 0;
};

//------------------------------------------------------------------------------------------------------------

// 
class IGradientTape : public virtual IObject {
public:
	// 
	virtual void Add( const CTapeBlob* result, const ITapeOperation* operation ) = 0;

	// 
	virtual void Remove( const CTapeBlob* result ) = 0;

	// 
	virtual CPtr<const ITapeOperation> GetOperation( const CTapeBlob* expression ) = 0;
	
};

} // namespace NeoML
