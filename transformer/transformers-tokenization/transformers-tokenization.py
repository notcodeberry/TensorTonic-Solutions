import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        words = set()

        for text in texts:
            for word in text.split():
                words.add(word)
        
        words = sorted(list(words))
        
        word_to_id = {word: i for i, word in enumerate(words)}

        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.word_to_id = {word: i for i, word in enumerate(special_tokens)}

        offset = len(self.word_to_id)

        for word, idx in word_to_id.items():
            self.word_to_id[word] = idx + offset
            
        self.id_to_word = {id: word for word, id in self.word_to_id.items()} 
        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.split()
        encoded = [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in words]
        
        """
        for word in text.split():
            if word in self.word_to_id:
                id = self.word_to_id[word]
            else: 
                id = self.word_to_id[self.unk_token]

            encoded.append(id)
        """
        
        return encoded 
            
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded = ' '.join([self.id_to_word[id] for id in ids])

        return decoded