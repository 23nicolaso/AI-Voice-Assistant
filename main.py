import requests
import playsound
import os
import whisper
import config
import sounddevice as sd
import soundfile as sf

while True:
    # Define constants
    CHANNELS = 1
    RATE = 16000
    DURATION = 6  # seconds
    OUTPUT_FILE = "output.mp3"

    # Record audio
    print("Recording audio for {} seconds...".format(DURATION))
    audio = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=CHANNELS)
    sd.wait()

    # Save audio to WAV file
    print("Saving audio to WAV file...")
    sf.write("output.wav", audio, samplerate=RATE)

    model = whisper.load_model("base")
    result = model.transcribe("output.wav", fp16=False)
    question = result["text"]
    print("Prompt:"+question)
    # Clean up the temporary file
    os.remove('output.wav')

    if "jarvis" in question.lower():
        # Define the payload with input messages and other parameters
        payload = {
            'messages': [
                {'role': 'system', 'content': 'You are an assistant for an IB Diploma Student. '
                                              'Answer all questions in 200 characters or less. Your name is Jarvis, '},
                {'role': 'user', 'content': question}
            ],
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 50
        }

        # Define the API endpoint and headers
        api_endpoint = 'https://api.openai.com/v1/chat/completions'
        api_key = config.OPENAI_API_KEY
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Send the API request
        response = requests.post(api_endpoint, headers=headers, json=payload)
        data = response.json()

        # Extract and display the output messages
        output_messages = data['choices'][0]['message']['content']
        print("Response ("+str(data['usage']['total_tokens'])+"): "+output_messages)

        eleven_load = {
          "text": output_messages,
          "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
          }
        }

        eleven_api_endpoint = 'https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM'
        eleven_api_key = config.ELEVENLABS_API_KEY  # Replace with your actual API key
        eleven_headers = {
            'xi-api-key': eleven_api_key
        }

        response = requests.post(eleven_api_endpoint, headers=eleven_headers, json=eleven_load)

        with open('temp.mp3', 'wb') as f:
            f.write(response.content)

        playsound.playsound('temp.mp3')

        # Clean up the temporary file
        os.remove('temp.mp3')