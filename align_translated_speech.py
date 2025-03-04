from transformers import AutoModel, AutoTokenizer
import itertools
import torch

# load model
model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")

# model parameters
align_layer = 8
threshold = 1e-3

# define inputs
src = """Tanzania, home to some of the most breathtaking wildlife on earth. Here in the heart of East Africa, 
    the great Serengeti National Park hosts one of nature's greatest spectacles, the Great Migration. 
    Over a million wildebeest, zebras, and gazelles travel vast distances in search of fresh grass, braving rivers filled with crocodiles. But predators are never far behind. 
    Lions, the kings of the savannah, stalk their prey with patience and precision. 
    Cheetahs, the fastest land animals, chase down their targets in a thrilling display of speed. 
    In the lush Terengair National Park, giant elephants roam freely. These intelligent creatures form strong family bonds, protecting their young from threats. 
    And in the ancient Ngorongoro crater, the endangered black rhino finds refuge, a rare sight in the wild. 
    Tanzania's wildlife is a treasure like no other, a delicate balance of nature that reminds us of the beauty and power of the wild."""

tgt = """
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

# pre-processing
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
  
# alignment
align_layer = 8
threshold = 1e-3
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
  
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

for i, j in sorted(align_words):
  print(f'{color.BOLD}{color.BLUE}{sent_src[i]}{color.END}==={color.BOLD}{color.RED}{sent_tgt[j]}{color.END}')


output_dict = {}
previous_word = None

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
        merged_dict[value] = [key]
# Step 2: Print the merged result
for key, value in merged_dict.items():
    print(f"{key} === {value}")
