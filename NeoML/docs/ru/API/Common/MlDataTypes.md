# Типы данных, используемые в машинном обучении

<!-- TOC -->

- [Типы данных, используемые в машинном обучении](#типы-данных-используемые-в-машинном-обучении)
    - [Вектор](#вектор)
        - [Класс CFloatVector](#класс-cfloatvector)
    - [Разреженный вектор](#разреженный-вектор)
        - [Описание CSparseFloatVectorDesc](#описание-csparsefloatvectordesc)
        - [Класс CSparseFloatVector](#класс-csparsefloatvector)
    - [Разреженная матрица](#разреженная-матрица)
        - [Описание CSparseFloatMatrixDesc](#описание-csparsefloatmatrixdesc)
        - [Класс CSparseFloatMatrix](#класс-csparsefloatmatrix)

<!-- /TOC -->

## Вектор

### Класс CFloatVector

Класс, реализующий стандартный (неразреженный) вектор с элементами типа `float`.

#### Конструкторы

```c++
CFloatVector(); // Конструктор по умолчанию. Используется, например, для конструирования объекта перед сериализацией значений из архива.
// Создать вектор размера size на основе разреженного, не заданные значения признаков устанавливаются в 0.
CFloatVector( int size, const CSparseFloatVector& sparseVector );
CFloatVector( int size, const CSparseFloatVectorDesc& sparseVector );
explicit CFloatVector( int size ); // Вектора размера size.
CFloatVector( int size, float init ); // Вектор размера size, каждый элемент которого равен init.
CFloatVector( const CFloatVector& other );
```

#### Операторы присваивания

```c++
CFloatVector& operator = ( const CFloatVector& vector );
CFloatVector& operator = ( const CSparseFloatVector& vector );
```

#### Методы

Конвертация стандартного вектора в разреженный с теми же значениями:

```c++
CSparseFloatVector SparseVector() const;
```

Проверка того, что вектор пуст:

```c++
bool IsNull() const;
```

Возвращает `true` в случаях, когда вектор был создан конструктором по умолчанию и после создания он не был считан из архива.

Получение числа элементов в векторе:

```c++
int Size() const;
```

Доступ к элементам:

```c++
float operator [] ( int i );
void SetAt( int i, float what );
```

Обнуление всех значений:

```c++
void Nullify();
```

Прибавление к вектору `vector * factor`:

```c++
CFloatVector& MultiplyAndAdd( const CFloatVector& vector, double factor );
CFloatVector& MultiplyAndAdd( const CSparseFloatVector& vector, double factor );
CFloatVector& MultiplyAndAdd( const CSparseFloatVectorDesc& vector, double factor );
```

Длина вектора (L2-норма):

```c++
double Norm() const;
```

#### Математические операции

```c++
CFloatVector& operator += ( const CFloatVector& vector );
CFloatVector& operator -= ( const CFloatVector& vector );
CFloatVector& operator *= ( double factor );
CFloatVector& operator /= ( double factor );
CFloatVector& operator = ( const CSparseFloatVector& vector );
CFloatVector& operator += ( const CSparseFloatVector& vector );
CFloatVector& operator -= ( const CSparseFloatVector& vector );
```

#### Сериализация

```c++
friend CArchive& operator << ( CArchive& archive, const CFloatVector& vector );
friend CArchive& operator >> ( CArchive& archive, CFloatVector& vector );
```

## Разреженный вектор

### Описание разреженного вектора CSparseFloatVectorDesc

Описание разреженного вектора, хранящее минимальную информацию, достаточную для извлечения данных. Не предоставляет возможность изменения данных.

```c++
struct NEOML_API CSparseFloatVectorDesc {
	int Size; // Количество аллоцированных элементов в векторе.
	int* Indexes; // Координаты в векторе.
	float* Values; // Значения в соответствующих координатах.
};
```

### Класс CSparseFloatVector

Разреженный вектор с элементами типа `float`. Не хранит свою длину.

#### Конструкторы

```c++
CSparseFloatVector();
explicit CSparseFloatVector( int bufferSize );
explicit CSparseFloatVector( const CSparseFloatVectorDesc& desc );
CSparseFloatVector( const CSparseFloatVector& other );
```

#### Оператор присваивания

```c++
CSparseFloatVector& operator = ( const CSparseFloatVector& vector );
```

#### Методы

Получение описания:

```c++
const CSparseFloatVectorDesc& GetDesc() const;
```

Количество аллоцированных элементов в векторе:

```c++
int NumberOfElements() const;
```

Доступ к элементам:

```c++
bool GetValue( int index, float& value ) const; // Возвращает true если элемент с координатой `index` аллоцирован.
float GetValue( int index ) const;
void SetAt( int index, float value );
```

Обнуление всех значений:

```c++
void Nullify();
```

Прибавление к вектору `vector * factor`:

```c++
CSparseFloatVector& MultiplyAndAdd( const CSparseFloatVector& vector, double factor );
```

Длина вектора (L2-норма):

```c++
double Norm() const;
```

#### Математические операции

```c++
CSparseFloatVector& operator += ( const CSparseFloatVector& vector );
CSparseFloatVector& operator -= ( const CSparseFloatVector& vector );
CSparseFloatVector& operator *= ( double factor );
CSparseFloatVector& operator /= ( double factor );
```

#### Сериализация

```c++
void Serialize( CArchive& archive );
```

## Разреженная матрица

### Описание CSparseFloatMatrixDesc

Описание разреженной матрицы, хранящее минимальную информацию, достаточную для извлечения данных. Не предоставляет возможности изменения данных.

```c++
// Описание разреженной матрицы.
struct NEOML_API CSparseFloatMatrixDesc {
	int Height; // Высота матрицы.
	int Width; // Ширина матрицы.
	int* Columns; // Указатель на колонки элементов.
	float* Values; // Указатель на значения элементов.
	int* PointerB; // Индексы начала данных векторов в Columns/Values.
	int* PointerE; // Индексы концов данных векторов в Columns/Values.

	// Получение описания строки в матрице.
	void GetRow( int index, CSparseFloatVectorDesc& desc ) const;
	CSparseFloatVectorDesc GetRow( int index ) const;
};
```

### Класс CSparseFloatMatrix

Разреженная матрица с элементами типа `float`.

#### Конструкторы

```c++
CSparseFloatMatrix() {}
CSparseFloatMatrix( int width, int rowsBufferSize = 0, int elementsBufferSize = 0 );
explicit CSparseFloatMatrix( const CSparseFloatMatrixDesc& desc );
CSparseFloatMatrix( const CSparseFloatMatrix& other );
```

#### Оператор присваивания

```c++
CSparseFloatMatrix& operator = ( const CSparseFloatMatrix& vector );
```

#### Методы

Получение описания:

```c++
const CSparseFloatMatrixDesc& GetDesc() const;
```

Получение размеров матрицы:

```c++
int GetHeight() const;
int GetWidth() const;
```

Добавление строки в матрицу:

```c++
void AddRow( const CSparseFloatVector& row );
void AddRow( const CSparseFloatVectorDesc& row );
```

Получение описания строки матрицы (т.е. разреженного вектора):

```c++
CSparseFloatVectorDesc GetRow( int index ) const;
void GetRow( int index, CSparseFloatVectorDesc& desc ) const;
```

#### Сериализация

```c++
void Serialize( CArchive& archive );
```
