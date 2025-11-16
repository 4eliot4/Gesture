from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import List
from .constants import GLOSSES
from .constants import MODEL_ID

"""
['america', 'angry', 'sick', 'sign language', 'speak', 'teach', 'thank you', 'tired', 'together', 'understand', 
'worry', 'backpack', 'yesterday', 'baseball', 'breakfast', 'difficult', 'dinner', 'enough', 'airplane', 'goodbye']
Is the list of glosses which are encoded into multiple tokens.
20 tokens, so 1/10th of our vocabulary.
"""

def encode_gloss(gloss: str,tok: PreTrainedTokenizerBase)->list[int]:
    """
    Encode a gloss into a list of IDs.
    """
    token_ids: List[int] = tok.encode(gloss, add_special_tokens=False)
    return token_ids


def get_multiple_tokens_glosses(glosses: list[str],tok: PreTrainedTokenizerBase)->list[str]:
    """
    Get the glosses which are encoded into multiple tokens.
    """
    
    encoded_glosses: list[list[int]] = [encode_gloss(gloss,tok) for gloss in glosses]
    encoded_glosses_multiple_tokens: list[str] = [gloss for gloss, ids in zip(glosses, encoded_glosses) if len(ids) > 1]
    return encoded_glosses_multiple_tokens

def count_multiple_tokens_glosses(glosses: list[str],tok: PreTrainedTokenizerBase)->int:
    """
    Count the number of glosses which are encoded into multiple tokens.
    """
    return len(get_multiple_tokens_glosses(glosses,tok))

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    encoded_glosses_multiple_tokens = get_multiple_tokens_glosses(GLOSSES,tok)
    print(encoded_glosses_multiple_tokens)