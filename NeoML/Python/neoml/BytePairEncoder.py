""" Copyright (c) 2017-2022 ABBYY Production LLC

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
import typing as tp

class BytePairEncoder():
    """BytePairEncoder trains byte pair encoding and
    encodes words using trained encoding"""

    def __init__(self):
        self._internal = PythonWrapper.BytePairEncoder()
        self.is_loaded = False
        
    def train(self, word_dictionary: tp.Dict[str, int], vocab_size: int, use_eow: bool=True, use_sow: bool=False) -> None:
        """Trains byte pair encoding.

        :param word_dictionary: a collection of words with count of times 
            each word appears in corpus.
        :type word_dictionary: dict-like of type str : int.

        :param vocab_size: maximum number of tokens in the encoder dictionary.
        :type vocab_size: int.
        
        :param use_eow: use special end-of-word token for each word in the word_dictionary
            and for each word that will be encoded after training is completed.
        :type use_eow: bool, default=True.
        
        :param use_sow: use special start-of-word token for each word in the word_dictionary
            and for each word that will be encoded after training is completed.
        :type use_sow: bool, default=False.

        :return: None
        """
        if self.is_loaded:
            raise ValueError("Encoder is already initialized.")
        self._internal.train(word_dictionary, vocab_size, use_eow, use_sow)
        self.is_loaded = True

    def load_from_dictionary(self, word_dictionary: tp.Dict[str, int], use_eow: bool=True, use_sow: bool=False) -> None:
        """Loads encoder from a dictionary with frequencies.

        :param word_dictionary: token dictionary with frequencies
        :type word_dictionary: dict-like of type str : int.
       
        :param use_eow: use special end-of-word token for each word in the word_dictionary
            and for each word that will be encoded after training is completed.
        :type use_eow: bool, default=True.
        
        :param use_sow: use special start-of-word token for each word in the word_dictionary
            and for each word that will be encoded after training is completed.
        :type use_sow: bool, default=False.
        """
        if self.is_loaded:
            raise ValueError("Encoder is already initialized.")
        self._internal.load_from_dictionary(word_dictionary, use_eow, use_sow)
        self.is_loaded = True

    def load(self, path: str) -> None:
        """Loads encoder from a binary file.

        :param path: path to file.
        : type path: str
        """
        if self.is_loaded:
            raise ValueError("Encoder is already initialized.")
        self._internal.load(path)
        self.is_loaded = True

    def store(self, path: str) -> None:
        """Saves encoder into a binary file.

        :param path: path to file.
        : type path: str
        """
        if not self.is_loaded:
            raise ValueError("Encoder is not initialized.")
        self._internal.store(path)
        
    def encode(self, text: tp.Union[str, tp.List[str]]) -> tp.Tuple[tp.List[int], tp.List[tp.Tuple[int, int]]]:
        """Encodes input text.

        :param text: input word or list of words.
        :type text: str or list of str

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
        if not self.is_loaded:
            raise ValueError("Encoder is not initialized.")
        if type(text) is str:
            text = [text]
        return self._internal.encode(text)

    def decode(self, token_ids: tp.List[int]) -> tp.List[str]:
        """Decodes sequence of token ids into sequence of words.

        :param token_ids: input sequence of token ids.
        :type token_ids: array-like of int.

        :return: sequence of words .
        :rtype: list of str.
        """
        if not self.is_loaded:
            raise ValueError("Encoder is not initialized.")
        return self._internal.decode(token_ids) 

    @property
    def size(self) -> tp.Optional[int]:
        """Returns the number of tokens in dictionary.
        :rtype: int.
        """
        if not self.is_loaded:
            return None
        return self._internal.get_size()

    @property
    def dictionary(self) -> tp.Optional[tp.Dict[str, int]]:
        """Returns the dictionary used by the encoder.
        :rtype: dict
        """
        if not self.is_loaded:
            return None
        return self._internal.get_dictionary()

    @property
    def use_eow(self) -> tp.Optional[bool]:
        """Returns End-Of-Word token usage flag
        :rtype: bool.
        """
        if not self.is_loaded:
            return None
        return self._internal.use_eow()
        
    @property
    def use_sow(self) -> tp.Optional[bool]:
        """Returns Start-Of-Word token usage flag
        :rtype: bool.
        """
        if not self.is_loaded:
            return None
        return self._internal.use_sow()

    @property
    def cache_period(self) -> tp.Optional[int]:
        """Returns the cache cleanup period. The cache is used for Encode calls acceleration.
        The result of the encode call is cached and will be erased if 
        no call with the same word will occur among next 1-2 X cachePeriod calls.
        :rtype: int.
        """
        if not self.is_loaded:
            return None
        return self._internal.get_cache_period()

    @cache_period.setter
    def cache_period(self, period: int) -> None:
        """Sets the cache cleanup period.
        """
        if not self.is_loaded:
            raise ValueError("Encoder is not initialized.")
        
        # -1 disables cache, 0 causes assert
        if period < 1:
            period = -1
        return self._internal.set_cache_period(period)
