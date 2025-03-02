from TTS.api import TTS

# Initialize the TTS engine with a pre-trained German model.
# Change the model name if a different German model is available.
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=True, gpu=False)

text = """
    Tansania, die Heimat einiger der atemberaubendsten Tierwelten der Erde. 
    Hier im Herzen Ostafrikas, beherbergt der große Serengeti-Nationalpark 
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
tts.tts_to_file(text=text, file_path="german_voice.wav")

print("Audio saved as german_voice.wav")

