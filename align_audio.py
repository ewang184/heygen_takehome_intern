import torch
import librosa
import soundfile as sf
from transformers import AutoModel, AutoTokenizer
import itertools
from speech_to_text_align import word_timings
from keybert import KeyBERT
import numpy as np
from librosa.effects import time_stretch
from librosa.core import resample

def get_text_keywords(text):
    kw_model = KeyBERT()
    kwds = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=25)

    word_list = []
    for w in kwds:
        word_list.append(w[0])
    return word_list

def stretch_audio(input_file, stretch_factor):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)

    # Apply time-stretching
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)

    # Save or plot the output if needed
    sf.write('stretched_audio.wav', y_stretched, sr)

    return y_stretched, sr

def align_words(src, tgt):
    src = src.replace(",", "").replace(".", "")
    tgt = tgt.replace(",", "").replace(".", "")

    model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
    tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")

    align_layer = 8
    threshold = 1e-3

    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

    output_dict = {}
    '''previous_word = None

    for word1, word2 in sorted(align_words):
        if sent_src[word1] != previous_word:
            output_dict[sent_src[word1]] = sent_tgt[word2]
        else:
            # If the current word1 is the same as the previous word1, append the new word2 to the existing value
            output_dict[sent_src[word1]] += sent_tgt[word2]
        previous_word = sent_src[word1]

    merged_dict = {}
    for key, value in output_dict.items():
        if value in merged_dict:
            merged_dict[value].append(key)  # Append the key to the existing list
        else:
            merged_dict[value] = [key]  # Create a new list with the key

    return merged_dict'''
    for word1, word2 in sorted(align_words):
        output_dict[sent_src[word1]] = sent_tgt[word2]
    return output_dict

def remove_crosses(pairs):
    # Sort pairs by the first element (ascending)
    pairs.sort(key=lambda x: x[0])
    
    # Initialize the result list
    result = []
    
    # Variable to keep track of the maximum `b` value seen so far
    max_b = float('-inf')
    
    # Traverse through the sorted list
    for a, b in pairs:
        # If the current `b` is greater than the max `b` seen so far, keep the pair
        if b > max_b:
            result.append((a, b))
            max_b = b  # Update max_b to the current `b`
    
    return result

def align_audio(audio_path_1, audio_path_2, timings, output_path):
    # Load the audio files
    audio_1, sr_1 = librosa.load(audio_path_1, sr=None)  # original audio 1
    audio_2, sr_2 = librosa.load(audio_path_2, sr=None)  # original audio 2
    
    # Resample to match sample rates
    if sr_1 != sr_2:
        audio_2 = librosa.resample(audio_2, orig_sr=sr_2, target_sr=sr_1)
        sr_2 = sr_1

    aligned_audio = np.zeros_like(audio_1)
    
    prev_start_eng = 0
    prev_start_de = 0

    for i,(start_eng, start_de) in enumerate(timings):
        start_eng_sample = int(start_eng * sr_1)
        end_eng_sample = int( * sr_1)
        
        # Extract the corresponding segment from the first audio
        segment_1 = audio_1[start_1_sample:end_1_sample]
        
        # Time-stretch the second audio segment to match the duration of the first segment
        start_2 = start_1  # matching start time
        end_2 = end_1      # matching end time
        segment_2, _ = librosa.load(audio_path_2, sr=sr_1, offset=start_2, duration=(end_2 - start_2))
        
        # Time-stretch the segment to match the length of segment_1
        stretch_factor = abs(len(segment_1) / len(segment_2))
        segment_2_stretched = time_stretch(segment_2, rate=stretch_factor)
        
        # Crossfade between the original aligned audio and the stretched segment_2
        fade_duration = 0.05  # duration of crossfade in seconds
        fade_samples = int(fade_duration * sr_1)
        
        if len(segment_2_stretched) > fade_samples:
            # Apply fade-in/out to segment_2_stretched
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            segment_2_stretched[:fade_samples] *= fade_in
            segment_2_stretched[-fade_samples:] *= fade_out
        
        # Add the adjusted segment to the aligned_audio
        aligned_audio[start_1_sample:end_1_sample] += segment_2_stretched[:len(segment_1)]

    # Write the aligned audio to a file
    sf.write(output_path, aligned_audio, sr_1)
    print(f"Aligned audio saved to {output_path}")

text_a = """Tanzania, home to some of the most breathtaking wildlife on earth. Here in the heart of East Africa, 
    the great Serengeti National Park hosts one of nature's greatest spectacles, the Great Migration. 
    Over a million wildebeest, zebras, and gazelles travel vast distances in search of fresh grass, braving rivers filled with crocodiles. But predators are never far behind. 
    Lions, the kings of the savannah, stalk their prey with patience and precision. 
    Cheetahs, the fastest land animals, chase down their targets in a thrilling display of speed. 
    In the lush Terengair National Park, giant elephants roam freely. These intelligent creatures form strong family bonds, protecting their young from threats. 
    And in the ancient Ngorongoro crater, the endangered black rhino finds refuge, a rare sight in the wild. 
    Tanzania's wildlife is a treasure like no other, a delicate balance of nature that reminds us of the beauty and power of the wild."""
text_b = """
    Tansania, die Heimat einiger der atemberaubendsten Tierwelten der Erde. Hier im Herzen Ostafrikas, beherbergt der große Serengeti-Nationalpark 
    eines der größten Spektakel der Natur, die Große Migration. Über eine Million Wildebeest, 
    Zebras und Gazellen reisen weite Strecken auf der Suche nach frischem Gras, 
    glühende Flüsse voller Krokodile. Aber Raubtiere sind nie weit dahinter. Löwen, 
    die Könige der Savanne, verfolgen ihre Beute mit Geduld und Präzision. Cheetahs, 
    die schnellsten Landtiere, jagen ihre Ziele in einer spannenden Schau der Geschwindigkeit. 
    Im üppigen Terengair-Nationalpark streifen riesige Elefanten frei umher. Diese intelligenten 
    Kreaturen bilden starke Familienbande, die ihre Jungen vor Bedrohungen schützen. Und im alten 
    Krater Ngorongoro findet der gefährdete schwarze Nashorn Zuflucht, ein seltener Anblick in der Wildnis. 
    Tansanias Wildtiere sind ein Schatz wie kein anderer, ein zartes Gleichgewicht der Natur, 
    das uns an die Schönheit und Kraft der Wildnis erinnert.
    """

english_word_timings = word_timings(text_a, "output_audio.wav")
german_word_timings = word_timings(text_b, "german_voice.wav")
word_dictionary = align_words(text_a, text_b)
important_words = get_text_keywords(text_a)

print(english_word_timings)
print(german_word_timings)
print(word_dictionary)
print(important_words)

important_words.append("-'-'-")
time_pairs = []
for word in important_words:
    print(f"word is {word}")
    print(f"capital word is {word.capitalize()}")

    try:
        german_word = word_dictionary[word]
    except KeyError:
        try:
            german_word = word_dictionary[word.capitalize()]
        except KeyError:
            print("German word doesn't exist in word alignment dictionary")
            german_word = ""
    i = 1
    while ((word.upper(), i) in english_word_timings):
        english_time = english_word_timings[(word.upper(), i)]
        try:
            german_time = german_word_timings[(german_word.upper(), i)]
            print(f"appending {(word, i), (german_word, i)}")
            time_pairs.append((english_time, german_time))
            i += 1
        except KeyError:
            print("not appending")
            i += 1
            continue

threshold = 6
close_time_pairs = [(a, b) for a, b in time_pairs if abs(a - b) <= threshold]
non_crossing_time_pairs = remove_crosses(close_time_pairs)
print(non_crossing_time_pairs)

align_audio('output_audio.wav', 'german_voice.wav', non_crossing_time_pairs, 'time_adjusted_german.wav')
