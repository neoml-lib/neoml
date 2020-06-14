# Общие классы

<!-- TOC -->

- [Общие классы](#общие-классы)
    - [Базовые классы](#базовые-классы)
        - [Массив CArray](#массив-carray)
        - [Строка CString](#строка-cstring)
        - [Подсчет ссылок IObject и CPtr](#подсчет-ссылок-iobject-и-cptr)
    - [Исключения](#исключения)
        - [CException](#cexception)
        - [CInternalError](#cinternalerror)
        - [CCheckException](#ccheckexception)
        - [CFileException](#cfileexception)
        - [CMemoryException](#cmemoryexception)
    - [Сериализация](#сериализация)
        - [CArchiveFile](#carchivefile)
            - [Создание и открытие файла](#создание-и-открытие-файла)
            - [Работа с позицией в файле](#работа-с-позицией-в-файле)
            - [Запись и чтение из файла](#запись-и-чтение-из-файла)
            - [Закрытие файла](#закрытие-файла)
        - [CArchive](#carchive)
            - [Создание и открытие архива](#создание-и-открытие-архива)
            - [Сериализация версии архива](#сериализация-версии-архива)
            - [Сериализация базовых типов](#сериализация-базовых-типов)

<!-- /TOC -->

## Базовые классы

В силу причин, изложенных в [руководстве](../../Introduction/README.md), библиотека **NeoML** имеет собственную реализацию некоторых базовых классов.

### Массив CArray

Собственная реализация массива называется `CArray`:

```c++
template<class T, class Allocator = CurrentMemoryManager>
class CArray;
```

Массив предоставляет методы доступа к элементам, добавления и удаления элементов, копирования массивов, а также сортировки и поиска.

Массив является шаблонным классом с двумя параметрами шаблона:

* тип хранимых в массиве объектов;
* специальный класс, определяющий механизм выделения и освобождения памяти под буфер массива.

По умолчанию второй параметр содержит класс `CurrentMemoryManager`, использующий операторы `new`/`delete`.

### Строка CString

Собственная реализация строки называется `CString` и отличается от стандартной строки из `C++` только дополнительным оператором приведения типа.

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

### Подсчет ссылок IObject и CPtr

Также в библиотеке реализован собственный механизм подсчета ссылок на объекты, выделенные оператором `new`.
Для его использования класс должен наследоваться от интерфейса `IObject`, а все операции с указателями на него должны производиться через "умный" указатель `CPtr`.

```c++
class FINE_CLASS IObject {
public:
	// Получение текущего значения счетчика ссылок.
	int RefCount() const;

	virtual void Serialize( CArchive& );

protected:
	IObject();
	virtual ~IObject();
};
```

Владеющий указатель `CPtr` гарантирует, что объект не будет разрушен, пока на него указывает хотя бы один `CPtr`.

```c++
template<class T>
class CPtr
```

Класс `CPtr` имеет полный набор методов для прозрачной работы с владеемым указателем.

## Исключения

В различных ситуациях функции и методы классов библиотеки могут бросать исключения нескольких типов.

### CException

```c++
typedef std::exception CException;
```

Базовый класс, от которого наследуются все исключения.

### CInternalError

```c++
typedef std::logic_error CInternalError;
```

Внутренняя ошибка. Также может возникнуть при передаче заведомо неверных параметров или при возникновении ошибочных ситуаций внутри библиотеки.

### CCheckException

```c++
typedef std::logic_error CCheckException;
```

Ошибка, возникшая из-за неверных параметров, которую невозможно было обнаружить в момент установки. Обычно это происходит в случаях, когда параметры операции соответствуют очевидным ограничениям, однако в дальнейшем приводят к некорректным ситуациям.

Вот лишь некоторые примеры:

* некорректное считывание из `CArchive`, хотя файл на диске существует и доступен на чтение;
* при использовании нейронных сетей какой-либо слой получил на вход данные неправильного типа или размера.

### CFileException

```c++
typedef std::system_error CFileException;
```

Ошибка работы с [файлами](#carchivefile).

### CMemoryException

```c++
typedef std::bad_alloc CMemoryException;
```

Ошибка работы с памятью.

## Сериализация

Библиотека **NeoML** использует собственный формат сериализации для моделей, нейросетей и т.п.

Сериализация в собственный формат реализуется двумя основными механизмами: абстрактным бинарным файлом и архивом, работающим с ним.
Абстрактный бинарный файл описывается интерфейсом `CBaseFile`, библиотека имеет универсальную реализацию этого интерфейса `CArchiveFile`.
Архив же предоставляет более высокоуровневый интерфейс для записи в этот файл базовых типов. Он реализован классом `CArchive`.

### CArchiveFile

Класс `CArchiveFile` представляет собой универсальный механизм чтения и записи бинарных файлов, работающий на всех платформах, где работает библиотека. В частности, он позволяет читать файлы из ресурсов приложения на Android.

#### Создание и открытие файла

Открыть или создать файл можно следующими конструкторами и методами:

```c++
CArchiveFile( const char* fileName, CArchive::TDirection direction, void* platformEnv = nullptr );
// или
CArchiveFile();
void Open( const char* fileName, CArchive::TDirection direction, void* platformEnv = nullptr );
```

Параметры:

* *fileName* — имя файла;
* *direction* должен быть `CArchive::load` для чтения из файла, и `CArchive::store` для записи в файл;
* *platformEnv* используется для передачи `AAssetManager*` при работе на `Android`.

После открытия текущая позиция в файле установлена на начало.

#### Работа с позицией в файле

Для работы с текущей позицией в файле существует следующий набор методов:

```c++
// Получение текущей позиции.
__int64 GetPosition() const;
int GetPosition32() const; // Только для файлов менее 2Gb.
// Установка текущей позиции.
__int64 Seek( __int64 offset, TSeekPosition from );
int Seek32( int offset, TSeekPosition from ); // Только для файлов менее 2Gb.
void SeekToBegin();
void SeekToEnd();
```

#### Запись и чтение из файла

Чтение из текущей позиции в файле `bytesCount` байтов и запись их в `buffer`. Возвращает количество считанных байтов.

```c++
int Read( void* buffer, int bytesCount );
```

Запись в файл `bytesCount` байтов из `buffer`:

```c++
void Write( const void* buffer, int bytesCount );  
```

Запись текущего содержимого файла на диск:

```c++
void Flush();
```

#### Закрытие файла

Для закрытия файла используется метод `Close()`.

После закрытия можно использовать этот же объект для открытия другого файла и работы с ним.

```c++
void Close();
```

### CArchive

Класс `CArchive` реализует архив, построенный над заданным бинарным файлом.

#### Создание и открытие архива

Для создания или открытия существующего архива используется следующий конструктор или метод `Open`:

```c++
CArchive( CBaseFile* baseFile, TDirection direction );

CArchive();
void Open( CBaseFile* baseFile, TDirection direction );
```

Параметры:

* *baseFile* — можно использовать `CArchiveFile` или реализовать собственный объект;
* *direction* может быть `CArchive::SD_Loading` для загрузки или `CArchive::SD_Storing` для записи. Для корректной работы необходим доступ к `baseFile` на чтение и запись соответственно.

#### Сериализация версии архива

Рекомендуется у каждого сериализуемого объекта иметь версию.

Для работы с версиями объектов существуют следующие методы:

```c++
int SerializeVersion( int currentVersion );
int SerializeVersion( int currentVersion, int minSupportedVersion );
```

Оба метода выполняют сериализацию версии; второй метод также проверяет, что версия в архиве не ниже минимальной поддерживаемой. Если версия в архиве ниже минимально поддерживаемой, метод бросает исключение `CCheckException`.

#### Сериализация базовых типов

Для сериализации базовых типов в архиве переопределены операторы `<<` и `>>`.

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
