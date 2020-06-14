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

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/NeoMathEngineDefs.h>

#if FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_ANDROID )

#include <PerformanceCountersCpuLinux.h>

#include <linux/perf_event.h>
#include <chrono>
#include <unistd.h>
#include <sys/syscall.h>

namespace NeoML {

struct CCounterInfo {
	decltype(perf_event_attr::type) Type;
	decltype(perf_event_attr::config) Config;
	const char* Name;
};

static const CCounterInfo counterInfo[] = {
	{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cpu_cycles"},
	{PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "cpu_instructions"},
	{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, "cache_acess"},
	{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "cache_miss"},
	{PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branch_count"},
	{PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branch_miss"},
	{PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES, "bus_cycles"},

	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK, "cpu_clock"},
	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "task_clock"},
	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, "page_faults"},
	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "context_switches"},
	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, "cpu_migrations"},
	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN, "page_faults_min"},
	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ, "page_faults_maj"},
	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_ALIGNMENT_FAULTS, "alig_faults"},
	{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_EMULATION_FAULTS, "emulations_faults"},
};

CPerformanceCountersCpuLinux::CPerformanceCountersCpuLinux() :
	IPerformanceCounters( counter ) 
{
	perf_event_attr conf;
	memset(&conf, 0, sizeof(struct perf_event_attr));
	conf.size = sizeof(struct perf_event_attr);
	conf.exclude_kernel = 1;
	conf.exclude_hv = 1;

	counter[0].Name = "time ms";
	counter[0].Value = 0;
	info[0].Fd = -1;
	info[0].Old = 0;
	CounterCount() = 1;

	static_assert( sizeof(counterInfo) / sizeof(*counterInfo) + 1 < MaxCounterCount, "countof(counterInfo) > MaxCounterCount" );

	for( size_t i = 0; i < sizeof(counterInfo) / sizeof(*counterInfo); ++i ) {
		conf.type = counterInfo[i].Type;
		conf.config = counterInfo[i].Config;
		int fd = syscall(__NR_perf_event_open, &conf, 0, -1, -1, 0);
		if( fd < 0 ) {
			continue;
		}

		counter[CounterCount()].Name = counterInfo[i].Name;
		counter[CounterCount()].Value = 0;
		info[CounterCount()].Fd = fd;
		info[CounterCount()].Old = 0;
		CounterCount()++;
	}
}

CPerformanceCountersCpuLinux::~CPerformanceCountersCpuLinux()
{
	for( size_t i = 0; i < CounterCount(); ++i ) {
		if( info[i].Fd >= 0 ) {
			close( info[i].Fd );
		}
	}
}

void CPerformanceCountersCpuLinux::Synchronise()
{
	auto cnow = std::chrono::steady_clock::now().time_since_epoch();
	auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(cnow).count();
	counter[0].Value = ( now - info[0].Old ) / 1000000;
	info[0].Old = now;

	for( size_t i = 1; i < CounterCount(); ++i ) {
		uint64_t value = 0;
		size_t len = read(info[i].Fd, &value, sizeof(value));
		if( len >= sizeof(value) ) {
			counter[i].Value = value - info[i].Old;
			info[i].Old = value;
		} else {
			counter[i].Value = 0;
			info[i].Old = 0;
		}
	}
}

} // namespace NeoML

#endif // #if FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_ANDROID )
