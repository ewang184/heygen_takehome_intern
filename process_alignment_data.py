# Example sentences
english_sentence = """Tanzania, home to some of the most breathtaking wildlife on earth. Here in the heart of East Africa, 
the great Serengeti National Park hosts one of nature's greatest spectacles, the Great Migration. 
Over a million wildebeest, zebras, and gazelles travel vast distances in search of fresh grass, braving rivers filled with crocodiles. But predators are never far behind. 
Lions, the kings of the savannah, stalk their prey with patience and precision. 
Cheetahs, the fastest land animals, chase down their targets in a thrilling display of speed. 
In the lush Terengair National Park, giant elephants roam freely. These intelligent creatures form strong family bonds, protecting their young from threats. 
And in the ancient Ngorongoro crater, the endangered black rhino finds refuge, a rare sight in the wild. 
Tanzania's wildlife is a treasure like no other, a delicate balance of nature that reminds us of the beauty and power of the wild."""

german_sentence = """Tansania, Heimat einiger der beeindruckendsten Wildtiere der Erde. Hier im Herzen von Ostafrika beherbergt der große Serengeti-Nationalpark eines der größten Naturschauspiele der Natur, die Große Migration. 
Mehr als eine Million Gnus, Zebras und Gazellen ziehen weite Strecken auf der Suche nach frischem Gras, wobei sie Flüsse überqueren, die mit Krokodilen gefüllt sind. Doch die Raubtiere sind nie weit entfernt. 
Löwen, die Könige der Savanne, schleichen geduldig und präzise auf ihre Beute zu. 
Geparden, die schnellsten Landtiere, jagen ihre Ziele in einer atemberaubenden Demonstration von Geschwindigkeit. 
Im üppigen Terengair-Nationalpark ziehen riesige Elefanten frei umher. Diese intelligenten Tiere bilden starke Familienbande und schützen ihren Nachwuchs vor Bedrohungen. 
Und im alten Ngorongoro-Krater findet das gefährdete schwarze Nashorn Zuflucht, ein seltener Anblick in der Wildnis. 
Die Tierwelt Tansanias ist ein Schatz wie kein anderer, ein empfindliches Gleichgewicht der Natur, das uns an die Schönheit und Kraft der Wildnis erinnert."""

# Function to prepare data in the required format
def prepare_data_for_alignment(src_text, tgt_text, output_file):
    # Split the sentences by periods (.)
    src_sentences = src_text.split(". ")
    tgt_sentences = tgt_text.split(". ")

    # Ensure that both lists of sentences are of the same length
    min_len = min(len(src_sentences), len(tgt_sentences))
    src_sentences = src_sentences[:min_len]
    tgt_sentences = tgt_sentences[:min_len]

    # Write the data in the required format to the output file
    with open(output_file, 'w') as f:
        for src, tgt in zip(src_sentences, tgt_sentences):
            f.write(f"{src.strip()} ||| {tgt.strip()}\n")

# Prepare the data and write it to a file
output_file = "data.txt"
prepare_data_for_alignment(english_sentence, german_sentence, output_file)

print(f"Data prepared and saved to {output_file}")

