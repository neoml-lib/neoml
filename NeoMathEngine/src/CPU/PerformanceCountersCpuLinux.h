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

// Class for aggregation performance statistics on Linux/Android
// This class use perf_event_open Linyx syscall
// This kernel mechanism can have different restrictions
// Current level stores in virtual file /proc/sys/kernel/perf_event_paranoid:
// -1  no restrictions.
//  0  allow access to CPU-specific data but not raw trace
//  1  allow both kernel and user measurements
//  2  allow only user-space measurements
//  3  not allowed? (not found full info)
//
// This class can work on 1 level
//
// For check current paranoid-level use:
// Linux: cat /proc/sys/kernel/perf_event_paranoid
// Android: adb shell cat /proc/sys/kernel/perf_event_paranoid
//
// For set 1 level use:
// Linux: sudo sysctl -w kernel.perf_event_paranoid=1
// Android: adb shell setprop security.perf_harden 0

#include <NeoMathEngine/PerformanceCounters.h>

namespace NeoML {

class CPerformanceCountersCpuLinux : public IPerformanceCounters {
public:
	CPerformanceCountersCpuLinux();
	~CPerformanceCountersCpuLinux() override;

	void Synchronise() override;

private:
	static const int MaxCounterCount = 32;

	struct CCounterInfo {
		CCounter::TCounterType Old;
		int Fd;
	};

	CCounter counter[MaxCounterCount];
	CCounterInfo info[MaxCounterCount];
};

} // namespace NeoML

