# Пример решения задачи кластеризации

<!-- TOC -->

- [Пример решения задачи кластеризации](#пример-решения-задачи-кластеризации)
	- [Подготовка входных данных](#подготовка-входных-данных)
	- [Реализация интерфейса данных для кластеризации](#реализация-интерфейса-данных-для-кластеризации)
	- [Запуск кластеризации](#запуск-кластеризации)
	- [Анализ результата](#анализ-результата)

<!-- /TOC -->

В этом примере мы используем библиотеку **NeoML** для решения задачи кластеризации данных на классическом датасете [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris). Будем использовать алгоритм *k-средних* (реализован классом [CKMeansClustering](../API/Clustering/kMeans.md)).

## Подготовка входных данных

Предположим, что входной датасет сериализован в файле в виде `CSparseFloatMatrix`. Используем средства сериализации библиотеки для загрузки данных.

```c++
CSparseFloatMatrix matrix;
CArchiveFile file( "iris.carchive", CArchive::load );
CArchive archive( &file, CArchive::load );
archive >> matrix;
```

## Реализация интерфейса данных для кластеризации

Все алгоритмы кластеризации получают на вход данные в виде интерфейса [IClusteringData](../API/Clustering/README.md); реализуем этот интерфейс над `CSparseFloatMatrix`.

```c++
class CClusteringData : public IClusteringData {
public:
	explicit CClusteringData( const CSparseFloatMatrix& _matrix ) :
		matrix( _matrix )
	{
	}

	virtual int GetVectorCount() const { return matrix.GetHeight(); }
	virtual int GetFeaturesCount() const { return matrix.GetWidth(); }
	virtual CSparseFloatVectorDesc GetVector( int index ) const { return matrix.GetRow( index ); }
	virtual CSparseFloatMatrixDesc GetMatrix() const { return matrix.GetDesc(); }
	virtual double GetVectorWeight( int /*index*/ ) const { return 1.0; }

private:
	CSparseFloatMatrix matrix;
};

CPtr<CClusteringData> data = new CClusteringData( matrix );
```

## Запуск кластеризации

Теперь, когда данные подготовлены, можно настроить алгоритм кластеризации. В классе `CParam` выставим настройки:

- *InitialClustersCount*: 3, т.к. в этом датасете три класса;
- *DistanceFunc*: `DF_Euclid`, чтобы использовать в виде функции расстояния евклидово расстояние между элементами;
- *MaxIterations* установим равным числу элементов в наборе данных.

```c++
CKMeansClustering::CParam params;
params.InitialClustersCount = 3;
params.DistanceFunc = DF_Euclid;	
params.MaxIterations = data->GetVectorCount();

CKMeansClustering kMeans( params );

CClusteringResult result;
kMeans.Clusterize( data, result );
```

## Анализ результата

Выведем результаты разбиения по классам.

```c++
printf("Count %d:\n", result.ClusterCount );
for( int i = 0; i < result.ClusterCount; i++ ) {
	for( int j = 0; j < result.Data.Size(); j++ ) {
		if( result.Data[j] == i ) {
			printf("%d ", j );
		}
	}
	printf("\n");
}
```

Как мы видим, алгоритм действительно разбил исходную выборку на три класса.

```
Count 3:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
50 51 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 101 106 113 114 119 121 123 126 127 133 138 142 146 149
52 77 100 102 103 104 105 107 108 109 110 111 112 115 116 117 118 120 122 124 125 128 129 130 131 132 134 135 136 137 139 140 141 143 144 145 147 148
```
