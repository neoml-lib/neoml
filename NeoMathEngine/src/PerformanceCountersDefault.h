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

#include <NeoMathEngine/PerformanceCounters.h>
#include <chrono>

namespace NeoML {

class CPerformanceCountersDefault : public IPerformanceCounters {
public:
	CPerformanceCountersDefault() : IPerformanceCounters( &counter )
	{
		counter.Name = "time ms";
		CounterCount() = 1;
	}

	void Synchronise() override
	{
		auto cnow = std::chrono::steady_clock::now().time_since_epoch();
		auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(cnow).count();
		counter.Value = ( now - old ) / 1000000;
		old = now;
	}

private:
	CCounter::TCounterType old;
	CCounter counter;
};

} // namespace NeoML

