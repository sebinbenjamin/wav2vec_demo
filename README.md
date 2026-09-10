# Speech Transcription using Wav2Vec 2.0

Welcome to the Speech Transcription project! This repository provides a solution for transcribing speech from WAV files using the powerful Wav2Vec 2.0 model. The pre-trained facebook/wav2vec2-large-xlsr-53 model supports multilingual speech recognition, making it versatile and effective for various languages.

## Features

- **Multilingual Support**: Transcribe speech with models covering 53 different languages.
- **Easy to Use**: Simple steps to set up and run.
- **Accurate Transcriptions**: Leverages advanced Wav2Vec 2.0 technology.
- **Audio Chunking**: Automatically splits long audio files into 1-minute chunks for processing.
- **Sample Rate Conversion**: Converts input audio to 16 kHz, the rate expected by Wav2Vec 2.0.
- **Pluggable Models**: Swap in any compatible Wav2Vec2 checkpoint via `--model_name`.
- **Progress Feedback**: Chunk processing is displayed with a progress bar (`tqdm`).

## How It Works

The pipeline is implemented in a single script, `transcribe.py`:

1. **Model and Processor**:
   - Uses `facebook/wav2vec2-large-xlsr-53` for multilingual support by default.
   - `Wav2Vec2Processor` handles audio preprocessing and decoding.
   - `Wav2Vec2ForCTC` is the CTC-fine-tuned model used for inference.

2. **Audio Loading and Chunking**:
   - Loads the audio file using `pydub`.
   - Converts the sample rate to 16 kHz.
   - Splits long audio files into 1-minute chunks, exported to `chunks/<filename>_chunk<N>.wav`.

3. **Preprocessing**:
   - Reads each chunk with `soundfile` and verifies it is sampled at 16 kHz.
   - Prepares the audio data to match the model's input requirements.

4. **Inference**:
   - Performs inference with `torch` (under `torch.no_grad()`) to get the logits.

5. **Decoding**:
   - Decodes the logits to produce the transcription text using greedy decoding (`torch.argmax` + `processor.decode`).
   - Saves all transcriptions to a text file.

## Repository Structure

```
.
├── transcribe.py        # The transcription pipeline
├── requirements.txt     # Python dependencies
├── chunks/              # Generated 1-minute WAV chunks (created at runtime)
├── transcriptions.txt   # Generated transcription output
└── wav2vec_env/         # Python virtual environment (git-ignored)
```

## Setup Instructions

Follow these simple steps to get started:

### 1. Clone the Repository

```bash
git clone https://github.com/sebinbenjamin/wav2vec2-transcription.git
cd wav2vec2-transcription
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv wav2vec_env
source wav2vec_env/bin/activate
```

### 3. Install System Dependencies

`pydub` relies on `ffmpeg` for decoding non-WAV formats:

```bash
sudo apt update
sudo apt install pkg-config libssl-dev build-essential ffmpeg
```

### 4. Install the Required Libraries

Upgrade pip and setuptools:

```bash
pip install --upgrade pip setuptools wheel
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

| Package | Purpose |
|---|---|
| `torch` | Model inference |
| `transformers` | Wav2Vec2 processor & model |
| `soundfile` | Reading WAV chunks |
| `pydub` | Resampling & chunk splitting |
| `tqdm` | Progress bar |
| `tokenizers` | Transformers tokenization backend |

### 5. Install Rust Compiler (if needed)

If the installation of `tokenizers` fails, install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Running the Script

### 1. Prepare Your WAV File

Ensure you have your WAV file ready (e.g., `default.wav`).

### 2. Run the Transcription Script

You can run the script with a specific WAV file:

```bash
python3 transcribe.py path/to/your/file.wav
```

If you want to use a different pre-trained model, specify the model name using the `--model_name` parameter:

```bash
python3 transcribe.py path/to/your/file.wav --model_name wav2vec2-large-xlsr-53-italian
```

If you don't provide any command-line arguments, the script will use `default.wav` as the input file and `facebook/wav2vec2-large-xlsr-53` as the model name:

```bash
python3 transcribe.py
```

### 3. View the Transcriptions

The transcriptions are saved to `transcriptions.txt`, with one section per chunk:

```
Chunk 0:
<transcribed text for the first minute>

Chunk 1:
<transcribed text for the second minute>
```

## Final Notes

- **Language Support**: The `facebook/wav2vec2-large-xlsr-53` model supports 53 languages, making it versatile for multilingual speech recognition. For best accuracy on a single language, consider a language-specific fine-tuned checkpoint via `--model_name`.
- **Sampling Rate**: Ensure your WAV file is sampled at 16 kHz for the best results, though the script will automatically convert the sample rate if necessary.
- **Error Handling**: If a chunk fails to read or infer, it is skipped and the remaining chunks continue processing.
- **Output Overwrite**: `transcriptions.txt` is overwritten on each run.

With this setup, you should be able to transcribe speech from a WAV file efficiently using the powerful Wav2Vec 2.0 model.

Happy transcribing!
