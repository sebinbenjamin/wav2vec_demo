# Speech Transcription using Wav2Vec 2.0

Welcome to the Speech Transcription project! This guide will help you use the pre-trained `facebook/wav2vec2-large-xlsr-53` model to transcribe speech from a WAV file in various languages.

## Features

- **Multilingual Support**: Transcribe speech in 53 different languages.
- **Easy to Use**: Simple steps to set up and run.
- **Accurate Transcriptions**: Leverages advanced Wav2Vec 2.0 technology.

## How It Works

1. **Model and Processor**:
   - Uses `facebook/wav2vec2-large-xlsr-53` for multilingual support.
   - `Wav2Vec2Processor` handles audio preprocessing and decoding.

2. **Audio Loading**:
   - Reads the WAV file using `soundfile`.
   - Ensures the sample rate is 16kHz.

3. **Preprocessing**:
   - Prepares the audio data to match the model's input requirements.

4. **Inference**:
   - Performs inference to get the logits.

5. **Decoding**:
   - Decodes the logits to produce the transcription text.

## Setup Instructions

Follow these simple steps to get started:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wav2vec2-transcription.git
cd wav2vec2-transcription

```


###  2. Create and Activate a Virtual Environment

```bash
python3 -m venv wav2vec_env
source wav2vec_env/bin/activate
```

###  3. Install the Required Libraries

```bash
pip install -r requirements.txt
```

## Running the Script

Once you have your environment set up, follow these steps:

#### 1. Prepare Your WAV File
Ensure you have your WAV file ready (e.g., sample.wav).

#### 2. Run the Transcription Script:
You can run the script with a specific WAV file and optionally specify the model name:
```bash
python3 transcribe.py path/to/your/file.wav
```

If you want to use a different pre-trained model, you can specify the model name using the --model_name parameter:

```bash
python3 transcribe.py path/to/your/file.wav --model_name your_model_name
```

If you don't provide any command-line arguments, the script will use sample.wav as the input file and facebook/wav2vec2-large-xlsr-53 as the model name.

```bash
python3 transcribe.py
```

## Final Notes

*Language Support*: The facebook/wav2vec2-large-xlsr-53 model supports 53 languages, making it versatile for multilingual speech recognition.
 
*Sampling Rate*: Ensure your WAV file is sampled at 16kHz for the best results.

With this setup, you should be able to transcribe speech from a WAV file efficiently using the powerful Wav2Vec 2.0 model.

Happy transcribing!