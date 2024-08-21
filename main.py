
import os
from pypdf import PdfReader 
import pyttsx3 

audios: list = ['.'.join(f.split('.')[:-1]) for f in os.listdir('audios')]
files: list = [f for f in os.listdir('books') if f.replace('.pdf', '') not in audios]

for file in files:
    path = open(os.path.join('books',file), 'rb') 
    if '.pdf' in file:
        pdfReader = PdfReader(path)
    
        # extracting the text from the PDF 
        text = ' '.join([page.extract_text() for page in pdfReader.pages])
    else:
        text = path.read()
        text = text.decode("utf-8")
    
    # reading the text 
    speak = pyttsx3.init() 
    voices = speak.getProperty("voices")
    speak.setProperty("voice", voices[0].id)
    speak.setProperty("rate", 150)
    speak.say(text)
    # speak.save_to_file(text=text.replace('\n', ' '), filename=os.path.join('audios', '.'.join(file.split('.')[:-1])+'.wav'))
    speak.runAndWait()









import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()

gen = tts.tts_with_preset(text=text.replace('\n', ' '))
for i, temp in enumerate(tts.tokenizer.preprocess_text(text.replace('\n', ' ')).split('.')[:5]):
    gen = tts.tts_with_preset(temp)
    torchaudio.save(os.path.join('audios', f'{i}_'+('.'.join(file.split('.')[:-1]))+'.wav'), gen.squeeze(0).cpu(), 24000)
    #IPython.display.Audio(os.path.join('audios', f'{i}_'+('.'.join(file.split('.')[:-1]))+'.wav'))










from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import random
import string
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
# load the processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# load the model
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
# load the vocoder, that is the voice encoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
# we load this dataset to get the speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

speakers = {
    'awb': 0,     # Scottish male
    'bdl': 1138,  # US male
    'clb': 2271,  # US female
    'jmk': 3403,  # Canadian male
    'ksp': 4535,  # Indian male
    'rms': 5667,  # US male
    'slt': 6799   # US female
}

def save_text_to_speech(text:str, filename:str, speaker:int=None) -> None:

    # preprocess text
    inputs = processor(text=text, return_tensors="pt").to(device)
    if speaker is not None:
        # load xvector containing speaker's voice characteristics from a dataset
        speaker_embeddings = torch.tensor(embeddings_dataset[speaker]["xvector"]).unsqueeze(0).to(device)
    else:
        # random vector, meaning a random voice
        speaker_embeddings = torch.randn((1, 512)).to(device)

    # generate speech with the models
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    if speaker is not None:
        # if we have a speaker, we use the speaker's ID in the filename
        output_filename = f"{filename}.mp3"
    else:
        # if we don't have a speaker, we use a random string in the filename
        random_str = ''.join(random.sample(string.ascii_letters+string.digits, k=5))
        output_filename = f"{filename}.mp3"
    # save the generated speech to a file with 16KHz sampling rate
    sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)
    


files: list = [f for f in os.listdir('books') if 'from-0-to-130' in f]

for file in files:
    path = open(os.path.join('books',file), 'rb') 
    
    pdfReader = PdfReader(path) 
    
    # the page with which you want to start 
    # this will read the page of 25th page. 
    pages = pdfReader.pages[15:] 
    
    # extracting the text from the PDF 
    text = ' '.join([page.extract_text() for page in pages])
    save_text_to_speech(text, filename=os.path.join('audios', '.'.join(file.split('.')[:-1])), speaker=speakers["bdl"])
