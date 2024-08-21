# Алгоритм ISODATA CIsoDataClustering

<!-- TOC -->

- [Алгоритм ISODATA CIsoDataClustering](#алгоритм-isodata-cisodataclustering)
	- [Параметры](#параметры)
	- [Пример](#пример)

<!-- /TOC -->

Метод представляет собой эвристический алгоритм, основанный на близости точек по геометрическому расстоянию. Описание можно найти в книге *Ball, Geoffrey H., Hall, David J.* Isodata: a method of data analysis and pattern classification. (1965)

Результат работы алгоритма во многом зависит от заданных начальных параметров.

В **NeoML** алгоритм реализован классом `CIsoDataClustering`, который предоставляет интерфейс `IClustering`. Кластеризация производится с помощью его метода `Clusterize`.

## Параметры

Параметры кластеризации описываются структурой `CIsoDataClustering::CParam`.

- *InitialClustersCount* — начальное количество кластеров;
- *MaxClustersCount* — максимальное количество кластеров, достигаемое во время работы алгоритма;
- *MinClusterSize* — минимальный размер кластера;
- *MaxIterations* — максимальное количество итераций;
- *MinClustersDistance* — минимальное расстояние между кластерами; если расстояние меньше, кластеры сливаются (*merge*);
- *MaxClusterDiameter* — максимальный диаметр кластера; если кластер больше, он может быть разбит на части (*split*);
- *MeanDiameterCoef* — коэффициент допустимого превышения среднего диаметра кластера; если диаметр кластера во столько раз превышает средний, он может быть разбит на части (*split*).

## Пример

В данном примере алгоритм ISODATA используется для кластеризации набора данных [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris):

```c++
void Clusterize( const IClusteringData& irisDataSet, CClusteringResult& result )
{
	CIsoDataClustering::CParam params;
	params.InitialClustersCount = 1;
	params.MaxClustersCount = 20;
	params.MinClusterSize = 1;
	params.MinClustersDistance = 0.60;
	params.MaxClusterDiameter = 1.0;
	params.MeanDiameterCoef = 0.5;
	params.MaxIterations = 50;

	CIsoDataClustering isoData( params );
	isoData.Clusterize( irisDataSet, result );
}
```