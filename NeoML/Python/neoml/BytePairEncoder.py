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

class BytePairEncoder:
    """BytePairEncoder trains byte pair encoding and
    encodes words using trained encoding"""

    def __init__(self, word_dictionary: tp.List[str], *,
                 eow: str, sow: str, 
                 use_raw_bytes: bool=False, unknown_token_id: int=0):
        """Creates byte pair encoder
        :param word_dictionary: A list of unique tokens ordered by order of merges during training (this is used when encoding).
        :type word_dictionary: list-like of type str.
       
        :param eow: special end-of-word token that will be added to each input word while encoding.
        :type eow: str.
        
        :param sow: special start-of-word token that will be added to each input word while encoding.
        :type sow: str.

        :param use_raw_bytes: treat strings as arrays of raw bytes (not utf-8).
        :type use_raw_bytes: bool, default=False.

        :param unknown_token_id: The id of the special 'unknown' token. All other tokens are enumerated from 'unknown_token_id' + 1.
        :type unknown_token_id: int, default=0.
        """
        self._internal = PythonWrapper.BytePairEncoder()
        
        if word_dictionary is None:
            # Special mode for creating with 'train'/'load'. Not intended to use externally.
            pass
        else:
            self._internal.initialize(word_dictionary, eow, sow, use_raw_bytes, unknown_token_id)
        
    @classmethod
    def train(cls, word_dictionary: tp.Dict[str, int], *, vocab_size: int, 
              use_eow: bool=True, use_sow: bool=False, 
              use_raw_bytes: bool=False, unknown_token_id: int = 0) -> "BytePairEncoder":
        """Trains byte pair encoding.

        :param word_dictionary: a collection of words with count of times each word appears in training corpus.
        :type word_dictionary: dict-like of type str : int.

        :param vocab_size: maximum number of tokens in the encoder dictionary.
        :type vocab_size: int.
        
        :param use_eow: use special end-of-word token for each word in the word_dictionary
            and for each word that will be encoded after training is completed.
        :type use_eow: bool, default=True.
        
        :param use_sow: use special start-of-word token for each word in the word_dictionary
            and for each word that will be encoded after training is completed.
        :type use_sow: bool, default=False.

        :return: BytePairEncoder
        """
        result = BytePairEncoder(None, eow=None, sow=None)
        result._internal.train(word_dictionary, vocab_size, use_eow, use_sow, use_raw_bytes, unknown_token_id)
        return result

    @classmethod
    def load(cls, path: str) -> "BytePairEncoder":
        """Loads encoder from a binary file.

        :param path: path to file.
        : type path: str
        """
        result = BytePairEncoder(None, eow=None, sow=None)
        result._internal.load(path)
        return result

    def store(self, path: str) -> None:
        """Saves encoder into a binary file.

        :param path: path to file.
        : type path: str
        """
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
        return self._internal.decode(token_ids) 

    @property
    def size(self) -> int:
        """Returns the number of tokens in dictionary.
        :rtype: int.
        """
        return self._internal.get_size()

    @property
    def dictionary(self) -> tp.Dict[str, int]:
        """Returns the dictionary used by the encoder.
        :rtype: dict
        """
        return self._internal.get_dictionary()

    @property
    def use_eow(self) -> bool:
        """Returns End-Of-Word token usage flag
        :rtype: bool.
        """
        return self._internal.use_eow()
        
    @property
    def use_sow(self) -> bool:
        """Returns Start-Of-Word token usage flag
        :rtype: bool.
        """
        return self._internal.use_sow()

    @property
    def use_raw_bytes(self) -> bool:
        """Returns use_raw_bytes flag
        :rtype: bool.
        """
        return self._internal.use_raw_bytes()

    @property
    def unknown_token_id(self) -> int:
        """Returns the id of the <UNK> token
        :rtype: bool.
        """
        return self._internal.unknown_token_id()

    @property
    def cache_period(self) -> int:
        """Returns the cache cleanup period. The cache is used for Encode calls acceleration.
        The result of the encode call is cached and will be erased if 
        no call with the same word will occur among next 1-2 X cachePeriod calls.
        :rtype: int.
        """
        return self._internal.get_cache_period()

    @cache_period.setter
    def cache_period(self, period: int) -> None:
        """Sets the cache cleanup period.
        """
        # -1 disables cache, 0 causes assert
        if period < 1:
            period = -1
        return self._internal.set_cache_period(period)
