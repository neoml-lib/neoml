/* Copyright © 2017-2022 ABBYY Production LLC

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

// Layer that multiplies some of the blob dimensions
// and fills the new element with approximated values based on its neighbors
// At the moment supports only linear interpolation
//
// By default each scale is equal to 1
class NEOML_API CInterpolationLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CInterpolationLayer )
public:
	explicit CInterpolationLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Rule types for dimensions
	enum class TRuleType : int {
		None, // Preserve dimension as-is
		Resize, // Change to fixed size (with interpolation)
		Scale, // Multiply dimension size (with interpolation)

		Count
	};

	// The rule for the given dimension
	struct CRule {
		CRule() : Type( TRuleType::None ), NewSize( NotFound ), ScaleCoeff( 1.f ) {}
		CRule( TRuleType type, int newSize, float scale ) : Type( type ), NewSize( newSize ), ScaleCoeff( scale ) {}

		static CRule Resize( int newSize ) { return CRule( TRuleType::Resize, newSize, 1.f ); }
		static CRule Scale( float scale ) { return CRule( TRuleType::Scale, NotFound, scale ); }

		void Serialize( CArchive& archive );

		TRuleType Type;
		int NewSize;
		float ScaleCoeff;
	};

	void SetRule( TBlobDim dim, const CRule& rule ) { NeoAssert( dim >= BD_BatchLength && dim < BD_Count ); rules[dim] = rule; }
	const CRule& GetRule( TBlobDim dim ) const { NeoAssert( dim >= BD_BatchLength && dim < BD_Count ); return rules[dim]; }

	// Sets the coordinate calculation mode
	// The variables in formula:
	//     - old_size - axis size before interpolation
	//     - new_size - axis size after interpolation
	//     - x_old - coordinate in array before the interpolation 
	//     - x_new - coordinate in array after the interpolation
	// The formulas:
	//     - TInterpolationCoords::Asymmetric    x_old = float( x_new * old_size ) / new_size
	//     - TInterpolationCoords::PytorchHalfPixel    x_old = new_size > 1 ? ( x_new + 0.5 ) * old_size / new_size - 0.5 : 0
	// By default TInterpolationCoords::Asymmetric
	TInterpolationCoords GetCoords() const { return coords; }
	void SetCoords( TInterpolationCoords newCoords ) { coords = newCoords; }

	// Sets the round mode (which transforms linear interpolation into nearest
	// Possible values:
	//     - TInterpolationRound::None - keep interpolation linear (*default)
	//     - TInterpolationRound::RoundPreferFloor
	//     - TInterpolationRound::RoundPreferCeil
	//     - TInterpolationRound::Floor
	//     - TInterpolationRound::Ceil
	TInterpolationRound GetRound() const { return round; }
	void SetRound( TInterpolationRound newRound ) { round = newRound; }

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	CArray<CRule> rules;
	TInterpolationCoords coords;
	TInterpolationRound round;
};

inline CArchive& operator<<( CArchive& archive, const CInterpolationLayer::CRule& rule )
{
	const_cast<CInterpolationLayer::CRule&>( rule ).Serialize( archive );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, CInterpolationLayer::CRule& rule )
{
	rule.Serialize( archive );
	return archive;
}

} // namespace NeoML
