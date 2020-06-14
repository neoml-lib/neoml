# Common Classes

<!-- TOC -->

- [Common Classes](#common-classes)
    - [Basic classes](#basic-classes)
        - [CArray](#carray)
        - [CString](#cstring)
        - [Reference counting: IObject and CPtr](#reference-counting-iobject-and-cptr)
    - [Exceptions](#exceptions)
        - [CException](#cexception)
        - [CInternalError](#cinternalerror)
        - [CCheckException](#ccheckexception)
        - [CFileException](#cfileexception)
        - [CMemoryException](#cmemoryexception)
    - [Serialization](#serialization)
        - [CArchiveFile](#carchivefile)
            - [Create and open files](#create-and-open-files)
            - [Position in file](#position-in-file)
            - [Read and write to files](#read-and-write-to-files)
            - [Close files](#close-files)
        - [CArchive](#carchive)
            - [Create and open archives](#create-and-open-archives)
            - [Serialize archive version](#serialize-archive-version)
            - [Serialize the basic types](#serialize-the-basic-types)

<!-- /TOC -->

## Basic classes

In **NeoML** library some of the basic classes have their own implementation. See the [introduction](../../Introduction/README.md) for an explanation of the reasons.

### CArray

The **NeoML** implementation of an array is called `CArray`:

```c++
template<class T, class Allocator = CurrentMemoryManager>
class CArray;
```

It provides the methods for accessing, adding and deleting the elements, copying the array, sorting and searching it.

The array is a template class with two parameters:

* the type of objects stored
* the memory management class

By default the second parameter is `CurrentMemoryManager`; it uses `new`/`delete` operations to allocate and release memory.

### CString

The **NeoML** implementation of a string is called `CString`. Its only difference from the standard C++ string is an additional type conversion operator:

```c++
class CString : public std::string {
public:
	CString() {}
	CString( const char* str ) : std::string( str ) {}
	CString( const char* str, int len ) : std::string( str, len ) {}
	CString( const std::string& str ) : std::string( str ) {}

	operator const char*() const { return data(); }
};
```

### Reference counting: IObject and CPtr

The library also implements its own reference counter for the objects allocated by the `new` operator.

The class that uses this reference counter should be derived from `IObject`, and `CPtr` smart pointer should be used for all operations with it.

```c++
class FINE_CLASS IObject {
public:
	// Gets the current reference count
	int RefCount() const;

	virtual void Serialize( CArchive& );

protected:
	IObject();
	virtual ~IObject();
};
```

The `CPtr` smart pointer guarantees that the object will not be destroyed while at least one `CPtr` points to it.

```c++
template<class T>
class CPtr
```

The `CPtr` class provides the full set of methods for working with the owned pointer.

## Exceptions

The functions and class methods of the **NeoML** library may throw various exceptions when errors occur.

### CException

```c++
typedef std::exception CException;
```

The base class for all exceptions.

### CInternalError

```c++
typedef std::logic_error CInternalError;
```

Internal library error. This exception is also thrown when unsupported parameters are passed.

### CCheckException

```c++
typedef std::logic_error CCheckException;
```

The error occurs because of incorrect parameters that were, however, not obviously wrong the moment they were set. This can happen when the parameters pass the basic check but then lead to conflicts.

Some examples:

* `CArchive` read error occurs, although the corresponding file exists and is accessible for reading.
* a layer in a neural network received input data of incorrect type or dimensions.

### CFileException

```c++
typedef std::system_error CFileException;
```

Error while working with [files](#carchivefile).

### CMemoryException

```c++
typedef std::bad_alloc CMemoryException;
```

Error while working with memory.

## Serialization

The **NeoML** library uses its own serialization for models, neural networks, etc.

The `CBaseFile` interface represents an abstract binary file. The library provides a universal implementation of this interface in the `CArchiveFile` class. The `CArchive` class provides a high-level interface for reading and writing basic data types into this file.

### CArchiveFile

The `CArchiveFile` class allows you to read and write into binary files on any platform supported by the library. This includes the files in Android app resources.

#### Create and open files

The following constructors and methods open or create files:

```c++
CArchiveFile( const char* fileName, CArchive::TDirection direction, void* platformEnv = nullptr );
// or
CArchiveFile();
void Open( const char* fileName, CArchive::TDirection direction, void* platformEnv = nullptr );
```

Parameters:

* *fileName* is the name of the file
* *direction* should be set to `CArchive::load` if you are going to read from this file and to `CArchive::store` if you are going to write
* *platformEnv* is used to pass the `AAssetManager*` pointer when working on Android

Once the file is open, the current position is at the start of the file.

#### Position in file

The following methods get and change the current position in file:

```c++
// Gets the current position
__int64 GetPosition() const;
int GetPosition32() const; // works only with files smaller than 2GB
// Sets the current position
__int64 Seek( __int64 offset, TSeekPosition from );
int Seek32( int offset, TSeekPosition from ); // works only with files smaller than 2GB
void SeekToBegin();
void SeekToEnd();
```

#### Read and write to files

```c++
int Read( void* buffer, int bytesCount );
```

Reads `bytesCount` bytes, starting with the current position in the file. The bytes are written into the `buffer`. The method returns the number of bytes actually read.

```c++
void Write( const void* buffer, int bytesCount );  
```

Writes `bytesCount` bytes from `buffer` into the file.

```c++
void Flush();
```

Saves the current contents of the file on disk.

#### Close files

Use the `Close()` method to close the open file. After closing you may use the same object to open another file and work with that.

```c++
void Close();
```

### CArchive

The `CArchive` class implements an archive over the specified binary file.

#### Create and open archives

The following constructors and methods create or open archives: 

```c++
CArchive( CBaseFile* baseFile, TDirection direction );

CArchive();
void Open( CBaseFile* baseFile, TDirection direction );
```

Parameters:

* *baseFile* — the file; use `CArchiveFile` or implement `CBaseFile` in your own class.
* *direction* should be set to `CArchive::SD_Loading` if you are going to read and to `CArchive::SD_Storing` if you are going to write. The corresponding read or write access to `baseFile` is required.

#### Serialize archive version

We recommend that you keep track of versions for each serializable object. The following methods work with object versions:

```c++
int SerializeVersion( int currentVersion );
int SerializeVersion( int currentVersion, int minSupportedVersion );
```

Both methods serialize the object version. The second method also checks that the version stored in the archive is at least the minimum supported version and throws the `CCheckException` if not.

#### Serialize the basic types

The `<<` and `>>`` operators are overloaded to serialize the basic types in the archive.

```c++
friend CArchive& operator <<( CArchive&, char variable );
friend CArchive& operator <<( CArchive&, signed char variable );
friend CArchive& operator <<( CArchive&, wchar_t variable );
friend CArchive& operator <<( CArchive&, bool variable );
friend CArchive& operator <<( CArchive&, short variable );
friend CArchive& operator <<( CArchive&, int variable );
friend CArchive& operator <<( CArchive&, __int64 variable );
friend CArchive& operator <<( CArchive&, float variable );
friend CArchive& operator <<( CArchive&, double variable );
friend CArchive& operator <<( CArchive&, unsigned char variable );
friend CArchive& operator <<( CArchive&, unsigned short variable );
friend CArchive& operator <<( CArchive&, unsigned int variable );
friend CArchive& operator <<( CArchive&, unsigned __int64 variable );
friend CArchive& operator >>( CArchive&, char& variable );
friend CArchive& operator >>( CArchive&, signed char& variable );
friend CArchive& operator >>( CArchive&, wchar_t& variable );
friend CArchive& operator >>( CArchive&, bool& variable );
friend CArchive& operator >>( CArchive&, short& variable );
friend CArchive& operator >>( CArchive&, int& variable );
friend CArchive& operator >>( CArchive&, __int64& variable );
friend CArchive& operator >>( CArchive&, float& variable );
friend CArchive& operator >>( CArchive&, double& variable );
friend CArchive& operator >>( CArchive&, unsigned char& variable );
friend CArchive& operator >>( CArchive&, unsigned short& variable );
friend CArchive& operator >>( CArchive&, unsigned int& variable );
friend CArchive& operator >>( CArchive&, unsigned __int64& variable );
```
