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

#include <NeoMathEngine/CrtAllocatedObject.h>
#include <cstdint>

namespace NeoML {

// Class for aggregating statistics
// All statistics can be look over range-based for
// All counters contain garbage after creation
// User must use Synchronise
class IPerformanceCounters : public CCrtAllocatedObject {
public:
	// Represent each performance counter
	struct CCounter {
		using TCounterType = uint_least64_t;

		const char* Name;
		TCounterType Value;
	};
	virtual ~IPerformanceCounters() = default;

	// Synchronise counters values
	// New values represent statistic since last Synchronise
	virtual void Synchronise() = 0;

	// Container methods
	size_t size() const { return counterCount; }
	const CCounter* begin() const { return counter; }
	const CCounter* end() const { return counter + counterCount; }
	const CCounter& operator[](size_t i) const { return counter[i]; }

protected:
	IPerformanceCounters( CCounter* _counter ) :
		counter(_counter),
		counterCount(0)
	{}

	size_t& CounterCount() { return counterCount; }

private:

	CCounter* counter;
	size_t counterCount;
};

} // namespace NeoML
