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

#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

// Calculates the variance for all features
void NEOML_API CalcFeaturesVariance( const IProblem& problem, CArray<double>& variance );

// Calculates the ratio of the variance over the whole set to the variance in one class
void NEOML_API CalcFeaturesVarianceRatio( const IProblem& problem, CArray<double>& varianceRatio );

// Calculates the chi-square statistic for all features
void NEOML_API CalcFeaturesChiSquare( const IProblem& problem, CArray<double>& chi2 ); 

// Calculates Pearson correlation for two features
double NEOML_API CalcTwoFeaturesCorrelation( const IProblem& problem, int index1, int index2 );

// Calculates Pearson correlation for a feature and a class
double NEOML_API CalcFeatureAndClassCorrelation( const IProblem& problem, int featureIndex, int classIndex );

// Calculates information gain for discrete features; for a continuous feature, 0 will be returned
void NEOML_API CalcFeaturesInformationGain( const IProblem& problem, CArray<double>& informationGain );

} // namespace NeoML
