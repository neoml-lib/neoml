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

namespace FObj {

typedef std::mutex CCriticalSection;

//---------------------------------------------------------------------------------------------------------------

class CCriticalSectionLock {
public:
	explicit CCriticalSectionLock( CCriticalSection& _section, bool initialLock = true );
	~CCriticalSectionLock();

	void Lock();
	void Unlock();
	bool IsLocked() const { return isLocked; }

private:
	CCriticalSection& section;
	bool isLocked;

	CCriticalSectionLock( const CCriticalSectionLock& );
	void operator=( const CCriticalSectionLock& );
};

inline CCriticalSectionLock::CCriticalSectionLock( CCriticalSection& _section, bool initialLock ) :
	section( _section ),
	isLocked( false )
{
	if( initialLock ) {
		Lock();
	}
}

inline CCriticalSectionLock::~CCriticalSectionLock()
{
	if( isLocked ) {
		Unlock();
	}
}

inline void CCriticalSectionLock::Lock()
{
	if( !isLocked ) {
		section.lock();
		isLocked = true;
	}
}

inline void CCriticalSectionLock::Unlock()
{
	if( isLocked ) {
		section.unlock();
		isLocked = false;
	}
}

} // namespace FObj
