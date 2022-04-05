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
    """BytePairEncoder trains byte pair encoding and
    encodes words using trained encoding"""

    def __init__(self):
        self.internal = PythonWrapper.BytePairEncoder()
        
    def train(self, word_dictionary, tokens_count, use_eow=True, use_sow=False):
        """Trains byte pair encoding.

        :param word_dictionary: the input word dictionary with count of times 
            each word appears in corpus.
        :type word_dictionary: dict-like of type str : int.

        :param tokens_count: the number of tokens in encoding.
        :type tokens_count: int.
        
        :param use_eow: use special end-of-word token for each word in the word_dictionary
            and for each word that will be encoded after training is completed.
        :type use_eow: bool, default=True.
        
        :param use_sow: use special start-of-word token for each word in the word_dictionary
            and for each word that will be encoded after training is completed.
        :type use_sow: bool, default=False.      

        :return: None
        """
        self.internal.train(word_dictionary, tokens_count, use_eow, use_sow)
        
    def encode(self, word):
        """Encodes input word.

        :param word: input word.
        :type word: str

        :return:
            - **token_ids** - ids of tokens.
                range of ids values = [-1, 0, ... , tokens_count].
                -1 is reserved value for unknown chars.
            - **token_positions** - starts and ends of tokens in the input word.
                the start is equal to the end for special tokens (eow, sow). 
        :rtype:
            - tuple(token_ids, token_positions)
            - **token_ids** - *list of token ids (int)*
            - **token_positions** - *list of pairs (int, int)*
        """
        return self.internal.encode(word)    
        
#-------------------------------------------------------------------------------------------------------------