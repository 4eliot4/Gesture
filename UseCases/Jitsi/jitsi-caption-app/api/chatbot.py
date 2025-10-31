#!/usr/bin/env python3
"""
Cheap Chat Completion with a few in‑context examples (few‑shot)

Usage:
    $ export OPENAI_API_KEY="sk-XXXXXX"
    $ python cheap_completion.py "Translate: Je suis fatigué"
"""
import os
import sys
import openai

# ----- 1. Configure ----------------------------------------------------------
MODEL = "gpt-4o-mini"        # cheapest chat model as of May 2025
TEMPERATURE = 0.8            # keep answers focused
MAX_TOKENS = 150             # cap response length

# ----- 2. Few‑shot examples ---------------------------------------------------
FEW_SHOT_MESSAGES = [
    # System rule – sets overall behaviour
    {"role": "system", "content": "You are a sign language translator"},

    # Example 1
    {"role": "user", "content": "Conclude from ['this', 'pay', 'learn', 'because', 'hello', 'yes'], ['this', 'pay'],['this', 'pay', 'place', 'animal'],['animal', 'call'],[think', 'understand', 'call', 'animal', 'who', 'me', 'yesterday', 'idea'],['who', 'yesterday', 'drink', 'understand'],['see', 'understand', 'deaf', 'drink', 'who', 'mother', 'yesterday']"},
    {"role": "assistant", "content": "Sentence: This is mother"},

    # Example 2
    {"role": "user", "content": "Conclude from '['sorry', 'think', 'deaf', 'country', 'love', 'understand', 'who', 'game', 'me'],['sorry', 'deaf', 'country', 'love', 'true', 'me'],['sorry', 'woman', 'sleep', 'country', 'true', 'me'],['see', 'understand', 'under', 'you', 'there'], ['see', 'not', 'this', 'understand', 'under', 'you', 'place', 'there'], ['see', 'this', 'understand', 'under', 'you', 'place', 'there'], ['deaf', 'understand', 'who', 'go', 'there'],['see', 'understand', 'under', 'you', 'there']"},
    {"role": "assistant", "content": "I understand "},

    # Example 3  (show it can handle short phrases too)
    {"role": "user", "content": "Conclude from ['under', 'true', 'must', 'you', 'place', 'buy', 'see', 'there', 'this'],['under', 'true', 'you', 'place', 'buy', 'see', 'there', 'this'],['under', 'true', 'you', 'place', 'see', 'there', 'this'],['go', 'watch', 'fast', 'place', 'camera', 'see'],['more', 'bed', 'idea', 'yesterday', 'day'], ['bed', 'more', 'understand', 'idea', 'yesterday', 'day'],['bed', 'more', 'teach', 'support', 'day'],['bed', 'more', 'country', 'family', 'house', 'support', 'day']"},
    {"role": "assistant", "content": "You go to bed "},
]

# ----- 3. Build request -------------------------------------------------------
def chat_complete(words_list: str) -> str:
    """
    Send `prompt` to the model together with the few‑shot examples and
    return the assistant's reply.
    """
    # Grab key once; raise clear error if not set
    api_key = ""
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY env‑var not found")
    
    user_prompt = f"Your job is to reconstruct the user's sentence from a sequence of words. For each word, you will receive a list of the likeliest possibilities for the current word. You will essentially receive a list of most possible words for each word in the sentence. Please keep in mind that this comes in from a live camera feed, so try to look for the sentence that makes the most sense. Conclude from this: {words_list}"

    client = openai.OpenAI(api_key=api_key)
    messages = FEW_SHOT_MESSAGES + [{"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def retrieve_sentence(words: list) -> str:
    sentence = ""
    return sentence

# ----- 4. Test execution ---------------------------------------------
if __name__ == "__main__":
    try:
        answer = chat_complete()
        print(answer)
    except Exception as exc:
        print(f"Error calling OpenAI: {exc}")