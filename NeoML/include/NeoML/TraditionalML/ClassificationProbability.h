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

namespace NeoML {

//	Classification probability
//	The probability takes values from 0 to 1
class NEOML_API CClassificationProbability {
public:
	//	Constructors
	CClassificationProbability();
	explicit CClassificationProbability( double value );

	//	Probability value
	double GetValue() const;
	void SetValue( double value );

	//	Comparison operators
	bool operator ==( const CClassificationProbability& classificationProbability ) const;
	bool operator !=( const CClassificationProbability& classificationProbability ) const;
	bool operator <( const CClassificationProbability& classificationProbability ) const;
	bool operator >( const CClassificationProbability& classificationProbability ) const;

	//	Transformation operators
	void operator &=( const CClassificationProbability& classificationProbability );
	void operator |=( const CClassificationProbability& classificationProbability );
	CClassificationProbability operator &( const CClassificationProbability& classificationProbability ) const;
	CClassificationProbability operator |( const CClassificationProbability& classificationProbability ) const;
	CClassificationProbability operator ~() const;

	//	Checks if the probability is valid
	bool IsValid() const;

	//	The minimum and maximum acceptable probability
	static const CClassificationProbability& MinProbability();
	static const CClassificationProbability& MaxProbability();

	//	Checks if the value is valid
	static bool IsValidValue( double value );

private:
	double value; // Probability value

	//	The minimum and maximum probability
	static const double minValue;
	static const double maxValue;
	static const CClassificationProbability minProbability;
	static const CClassificationProbability maxProbability;
	//	The value precision
	static const double precision;

	CClassificationProbability( double value, const void* dummy );

	void setValue( double value );
	static CClassificationProbability make( double value );
};

inline CArchive& operator <<( CArchive& archive, const CClassificationProbability& classificationProbability )
{
	archive << classificationProbability.GetValue();
	return archive;
}

inline CArchive& operator >>( CArchive& archive, CClassificationProbability& classificationProbability )
{
	double value;
	archive >> value;
	classificationProbability.SetValue( value );
	return archive;
}

inline CClassificationProbability::CClassificationProbability()
{
	setValue( minValue );
}

inline CClassificationProbability::CClassificationProbability( double src )
{
	SetValue( src );
}

inline double CClassificationProbability::GetValue() const
{
	return value;
}

inline void CClassificationProbability::SetValue( double src )
{
	NeoAssert( IsValidValue( src ) );

	setValue( src );
}

inline bool CClassificationProbability::operator ==( const CClassificationProbability& src ) const
{
	return value == src.value;
}

inline bool CClassificationProbability::operator !=( const CClassificationProbability& src ) const
{
	return !( *this == src );
}

inline bool CClassificationProbability::operator <( const CClassificationProbability& src ) const
{
	return value < src.value;
}

inline bool CClassificationProbability::operator >( const CClassificationProbability& src ) const
{
	return src < *this;
}

inline void CClassificationProbability::operator &=( const CClassificationProbability& src )
{
	*this = *this & src;
}

inline void CClassificationProbability::operator |=( const CClassificationProbability& src )
{
	*this = *this | src;
}

inline CClassificationProbability CClassificationProbability::operator &( const CClassificationProbability& src ) const
{
	return make( value * src.value );
}

inline CClassificationProbability CClassificationProbability::operator |( const CClassificationProbability& src ) const
{
	return make( value + src.value - value * src.value );
}

inline CClassificationProbability CClassificationProbability::operator ~() const
{
	return make( maxValue - value );
}

inline bool CClassificationProbability::IsValid() const
{
	return IsValidValue( value );
}

inline bool CClassificationProbability::IsValidValue( double value )
{
	if( value < minValue - precision ) {
		return false;
	}
	if( maxValue + precision < value ) {
		return false;
	}

	return true;
}

inline CClassificationProbability::CClassificationProbability( double src, const void* /*dummy*/ )
{
	setValue( src );
}

inline void CClassificationProbability::setValue( double src )
{
	value = src;
}

inline CClassificationProbability CClassificationProbability::make( double value )
{
	return CClassificationProbability( value, 0 );
}

} // namespace NeoML
