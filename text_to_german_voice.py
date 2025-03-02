from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.api import TTS


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

'''config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.cuda()

outputs = model.synthesize(
    text,
    config,
    speaker_wav="/data/TTS-public/_refclips/3.wav",
    gpt_cond_len=3,
    language="en",
)

print(outputs)'''

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

tts.tts_to_file(text="Tansania, die Heimat einiger der atemberaubendsten Tierwelten der Erde.",
                file_path="german_voice.wav",
                speaker_wav="output_audio.wav",
                language="de")



print("Audio saved as german_voice.wav")

