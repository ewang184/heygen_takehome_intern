import os
from transformers import MarianMTModel, MarianTokenizer

src_lang = "en"
tgt_lang = "de"
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def get_word_level_alignment(src_text, translated_text):
    # Tokenize source (English) and target (German) texts
    src_tokens = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True)
    tgt_tokens = tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True)

    src_token_ids = src_tokens["input_ids"][0].tolist()
    tgt_token_ids = tgt_tokens["input_ids"][0].tolist()

    # Convert token IDs to actual tokens
    src_tokens = tokenizer.convert_ids_to_tokens(src_token_ids)
    tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_token_ids)

    # Align the words (handle subwords by joining tokens into words)
    src_words = [token.replace("▁", "") for token in src_tokens]
    tgt_words = [token.replace("▁", "") for token in tgt_tokens]

    return src_words, tgt_words

def print_alignment(src_text, translated_text):
    src_words, tgt_words = get_word_level_alignment(src_text, translated_text)

    print(f"Source (English) text: {src_text}")
    print(f"Translated (German) text: {translated_text}")
    print("\nWord-level alignment:")

    # Print aligned words
    for src_word, tgt_word in zip(src_words, tgt_words):
        print(f"{src_word} <-> {tgt_word}")

if __name__ == "__main__":

    text = """Tanzania, home to some of the most breathtaking wildlife on earth. Here in the heart of East Africa, 
    the great Serengeti National Park hosts one of nature's greatest spectacles, the Great Migration. 
    Over a million wildebeest, zebras, and gazelles travel vast distances in search of fresh grass, braving rivers filled with crocodiles. But predators are never far behind. 
    Lions, the kings of the savannah, stalk their prey with patience and precision. 
    Cheetahs, the fastest land animals, chase down their targets in a thrilling display of speed. 
    In the lush Terengair National Park, giant elephants roam freely. These intelligent creatures form strong family bonds, protecting their young from threats. 
    And in the ancient Ngorongoro crater, the endangered black rhino finds refuge, a rare sight in the wild. 
    Tanzania's wildlife is a treasure like no other, a delicate balance of nature that reminds us of the beauty and power of the wild."""


    translated_text = translate_text(text)
    print(translated_text)

    print_alignment(text, translated_text)
