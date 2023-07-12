/* Copyright © 2017-2023 ABBYY

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

namespace NeoML {

namespace optimization {

class CGraph;

// Unpacks content of non-recurrent composites into the root CDnn
// Returns the number of unpacked composites
int UnpackComposites( CGraph& graph );

// Removes trivial layers (dropouts, linear(1,0) etc.)
// Returns the number of removed layers
int RemoveTrivialLayers( CGraph& graph );

} // namespace optimization

} // namespace NeoML
