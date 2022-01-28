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

// Gradient calculation engine.
// Implements automatic gradient calculation for all functions defined in AutoDiffFunctions.h,
// and also for user-defined functions.
class NEOML_API CGradientTape {
public:
	CGradientTape();
	~CGradientTape();

	// Creates a tape variable that will store the operations of the forward pass.
	CPtr<const CDnnBlob> Variable( const CDnnBlob& blob );

	// Computes the gradient using the operations recorded in this tape.
	CPtr<const CDnnBlob> Gradient( const CDnnBlob& expression, const CDnnBlob& var );

private:
	CPtr<CGradientTapeImpl> impl;

	CGradientTape( const CGradientTape& ) = delete;
	CGradientTape& operator=( const CGradientTape& ) = delete;
};

//------------------------------------------------------------------------------------------------------------

// The blob with the info needed for gradient calculation.
class NEOML_API CTapeBlob : public CDnnBlob {
public:
	CTapeBlob( IGradientTape* tape, const CDnnBlob& blob );
	CTapeBlob( IGradientTape* tape, IMathEngine& mathEngine, const CBlobDesc& desc );

	// Gets the used gradient calculation engine.
	CPtr<IGradientTape> Tape() const { return tape; }
	
protected:
	~CTapeBlob() override;

	// Detaches the blob from the tape.
	void Detach() const;

private:
	mutable CPtr<IGradientTape> tape;

	friend CGradientTapeImpl;
};

//------------------------------------------------------------------------------------------------------------

// The interface for a tape operation.
// If you wish to use autodifferentiation with CGradientTape for user-defined operations,
// all these operations should implement this interface.
class ITapeOperation : public IObject {
public:
	// Returns the jacobian of the 'var'.
	// Jacobian has size GetObjectCount() x GetObjectSize().
	// If GetObjectCount() == 1 then jacobian is a diagonal matrix of GetObjectSize() x GetObjectSize() size, stored like a vector.
	virtual CPtr<CDnnBlob> Jacobian( const CTapeBlob* var ) const = 0;
};

//------------------------------------------------------------------------------------------------------------

// Internal gradient calculation engine interface.
// This is where the user-defined operations get added to the tape.
class IGradientTape : public virtual IObject {
public:
	// Adds operation which has been used for calculating blob 'result'.
	virtual void Add( const CTapeBlob* result, const ITapeOperation* operation ) = 0;

	// Removes operation for the blob.
	virtual void Remove( const CTapeBlob* result ) = 0;

	// Gets operation which has been used for calculating blob 'expression'.
	virtual CPtr<const ITapeOperation> GetOperation( const CTapeBlob* expression ) = 0;
};

} // namespace NeoML
