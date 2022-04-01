""" Copyright (c) 2017-2021 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------
"""

import neoml.PythonWrapper as PythonWrapper

class BytePairEncoder() :
    def __init__(self):
        self.internal = PythonWrapper.BytePairEncoder()
        
    def build(self, words, iterations_count):
        self.internal.build(words, iterations_count)
        
    def encode(self, word):
        return self.internal.encode(word)    
        
    def decode(self, encoding):
        return self.internal.decode(encoding)
        
#-------------------------------------------------------------------------------------------------------------