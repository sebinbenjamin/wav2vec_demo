import argparse
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transcribe speech from a WAV file using Wav2Vec2.0.")
    parser.add_argument(
        "wav_file",
        type=str,
        nargs='?',
        default="default.wav",
        help="Path to the WAV file to be transcribed (default: 'default.wav')"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-large-xlsr-53",
        help="Name of the pretrained model to use (default: 'facebook/wav2vec2-large-xlsr-53')"
    )
    args = parser.parse_args()

    # Clear Hugging Face cache (optional but recommended if facing issues)
    from transformers import file_utils
    file_utils.hf_cache_home = "path/to/your/cache_directory"  # Optional: Specify a different cache directory
    file_utils.clean_cache()

    # Load pretrained model and processor for multilingual support
    try:
        processor = Wav2Vec2Processor.from_pretrained(args.model_name)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load the specified audio file
    try:
        audio_input, sample_rate = sf.read(args.wav_file)
    except Exception as e:
        print(f"Error reading the audio file: {e}")
        return

    # Ensure the audio is sampled at 16kHz as expected by the model
    if sample_rate != 16000:
        print(f"Expected sampling rate 16000, but got {sample_rate}")
        return

    # Preprocess the audio file
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    # Perform inference
    try:
        with torch.no_grad():
            logits = model(input_values).logits

        # Get predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the ids to text
        transcription = processor.decode(predicted_ids[0])
        print("Transcription:", transcription)
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
