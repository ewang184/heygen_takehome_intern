import torch
import torchaudio
from dataclasses import dataclass
import IPython
import matplotlib.pyplot as plt
from collections import defaultdict

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def convert_to_format(input_string):
    # Remove commas and split the input string by spaces
    input_string = input_string.replace(",", "")
    input_string = input_string.replace(".", " -'-'- ")

    german_to_english_map = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss',
        'Ä': 'Ae',
        'Ö': 'Oe',
        'Ü': 'Ue',
        'é': 'e',  # Typically phoneticized as 'e'
        'è': 'e',  # Typically phoneticized as 'e'
        'à': 'a',  # Typically phoneticized as 'a'
    }
    
    # Replace each German character in the string with its English phonetic counterpart
    for german_char, english_phonetic in german_to_english_map.items():
        input_string = input_string.replace(german_char, english_phonetic)

    words = input_string.split()
    # Capitalize every word and join them with "|"
    formatted_string = "|" + "|".join(word.upper() for word in words) + "|"
    return formatted_string


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def display_segment(i, waveform, bundle, word_segments, trellis):
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    #print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
    #segment = waveform[:, x0:x1]
    #return IPython.display.Audio(segment.numpy(), rate=bundle.sample_rate)
    return word.label, (x0 / bundle.sample_rate), (x0 / bundle.sample_rate)

def word_timings(text, wav_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.random.manual_seed(0)

    SPEECH_FILE = wav_file

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    with torch.inference_mode():
        waveform, _ = torchaudio.load(SPEECH_FILE)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    transcript = convert_to_format(text)
    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript]

    trellis = get_trellis(emission, tokens)

    path = backtrack(trellis, emission, tokens)
    for p in path:
        segments = merge_repeats(path, transcript)
    for seg in segments:

        word_segments = merge_words(segments)
    for word in word_segments:

        word_timings_list = []
    for i,word in enumerate(word_segments):
        word,start,end = display_segment(i, waveform, bundle, word_segments, trellis)
        word_timings_list.append((word,start,end))

    word_count = defaultdict(int)
    result = {}
    for word, time_start, time_end in word_timings_list:
        word_count[word] += 1
        result[(word, word_count[word])] = time_start

    return result

def extract_special_timings(data):
    special_timings = []
    for key, time in data.items():
        if key[0] == "-'-'-":  # Check if the key starts with "-'-'-"
            special_timings.append(time)
    return special_timings

text = """Tanzania, home to some of the most breathtaking wildlife on earth. Here in the heart of East Africa, 
    the great Serengeti National Park hosts one of nature's greatest spectacles, the Great Migration. 
    Over a million wildebeest, zebras, and gazelles travel vast distances in search of fresh grass, braving rivers filled with crocodiles. But predators are never far behind. 
    Lions, the kings of the savannah, stalk their prey with patience and precision. 
    Cheetahs, the fastest land animals, chase down their targets in a thrilling display of speed. 
    In the lush Terengair National Park, giant elephants roam freely. These intelligent creatures form strong family bonds, protecting their young from threats. 
    And in the ancient Ngorongoro crater, the endangered black rhino finds refuge, a rare sight in the wild. 
    Tanzania's wildlife is a treasure like no other, a delicate balance of nature that reminds us of the beauty and power of the wild."""

timings = word_timings(text, "output_audio.wav")
sentence_timings = extract_special_timings(timings)

print(sentence_timings)
