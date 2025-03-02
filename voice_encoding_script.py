import os
import glob
import numpy
import argparse
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)
    print(signal.shape)
    print("was signal shape")

    assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    #assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    embeddings = embeddings.reshape(1, 512)
    print(f"embed shape is {embeddings.shape}")
    return embeddings

def process(wav_path, spkemb, spkemb_root):

    if not os.path.exists(spkemb_root):
        print(f"Create speaker embedding directory: {spkemb_root}")
        os.mkdir(spkemb_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_embed = spkemb#"speechbrain/spkrec-xvect-voxceleb"
    classifier = EncoderClassifier.from_hparams(source=speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', speaker_embed))
    size_embed = spk_model[speaker_embed]
    
    utt_id = os.path.basename(wav_path).replace(".wav", "")  # Use file name as ID
    utt_emb = f2embed(wav_path, classifier, size_embed)
    output_file = os.path.join(spkemb_root, f"{utt_id}.npy")
    np.save(output_file, utt_emb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-file", "-i", required=True, type=str, help="Path to the .wav file.")
    parser.add_argument("--output-root", "-o", required=True, type=str, help="Directory to save the extracted embedding.")
    parser.add_argument("--speaker-embed", "-s", type=str, required=True, choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
                        help="Pretrained model for extracting speaker embedding.")
    
    args = parser.parse_args()

    print(f"Extracting speaker embedding from {args.wav_file}, "
          + f"Saving to {args.output_root}, "
          + f"Using speaker model {args.speaker_embed} with {spk_model[args.speaker_embed]} size.")
    
    process(args.wav_file, args.speaker_embed, args.output_root)

if __name__ == "__main__":
    """
    Example usage:
    python extract_single_embedding.py \
        -i /path/to/your/file.wav \
        -o /path/to/save/embedding \
        -s speechbrain/spkrec-xvect-voxceleb
    """
    main()
