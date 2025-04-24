# Microphone Transcription with Whisper API

This script records audio from your microphone and uses OpenAI's Whisper API to transcribe it to text.

## Setup

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. On macOS, you may need to install PortAudio first if PyAudio installation fails:
```
brew install portaudio
```

## Usage

Run the script:
```
python transcribe_mic.py
```

The script will:
1. Ask you how many seconds to record
2. Record audio from your microphone
3. Send the audio to OpenAI's Whisper API
4. Display the transcription

To exit the script, enter 'q' when prompted for recording duration.

## Requirements

- Python 3.7+
- OpenAI API key
- Microphone access 