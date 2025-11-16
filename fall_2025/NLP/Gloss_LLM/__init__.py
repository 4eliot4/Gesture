"""Gloss_LLM public API.

Expose commonly used constants and utility functions from the
`resized_vocabulary` and `gloss_encoding` modules so that callers can
import them from `NLP.Gloss_LLM`.
"""

from .constants import MODEL_ID, GLOSSES, DTYPE, DEVICE_MAP
from .resized_vocabulary import (
	allowed_sequences,
	allowed_token_ids,
	token_str,
	print_prob_table,
	compute_next_gloss_probability,
	compute_next_gloss_probability_batch,
	get_probability_of_a_gloss,
	get_probability_of_a_gloss_multitoken,
	compute_gloss_probabilities_for_slot,
	compute_gloss_probabilities_for_slot_batch,
	prompt_testing,
)

from .gloss_encoding import encode_gloss

__all__ = [
	"MODEL_ID",
	"GLOSSES",
	"DTYPE",
	"DEVICE_MAP",
	"allowed_sequences",
	"allowed_token_ids",
	"token_str",
	"print_prob_table",
	"compute_next_gloss_probability",
	"compute_next_gloss_probability_batch",
	"get_probability_of_a_gloss",
	"get_probability_of_a_gloss_multitoken",
	"compute_gloss_probabilities_for_slot",
	"compute_gloss_probabilities_for_slot_batch",
	"prompt_testing",
	"encode_gloss",
	"decode_gloss",
]
