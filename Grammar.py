import asyncio

import gradium
import sounddevice as sd
import speech_recognition as sr
from openai import OpenAI

groqKey = "key" 
gradiumKey = "key" 
voiceID = "key" 

groqClient = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groqKey)


def listen_and_transcribe():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0
    with sr.Microphone() as source:
        print("Calibrating...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Speak now: ")
        audio = recognizer.listen(source)

    print("Transcribing...")

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Retry, your voice was not understandable...")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return None


def fix_grammar(broken_text):
    print("Fixing grammar...")
    response = groqClient.chat.completions.create(
        model="llama-3.1-8b-versatile", 
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a real-time grammar corrector. Fix the user's sentence. "
                    "Output ONLY the corrected sentence. Do not include any intro, "
                    "explanations, or quotes."
                ),
            },
            {"role": "user", "content": broken_text},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


async def stream_voice_clone(text):
    print("Playing Audio...")
    client = gradium.client.GradiumClient(api_key=gradiumKey)

    stream = await client.tts_stream(setup={"voice_id": voiceID, "output_format": "pcm", }, text=text,)

    out_stream = sd.RawOutputStream(samplerate=48000, channels=1, dtype="int16")
    out_stream.start()
    try:
        async for chunk in stream.iter_bytes():
            out_stream.write(chunk)
    finally:
        out_stream.stop()
        out_stream.close()


if __name__ == "__main__":
    print("--- Grammar Fixer ---")

    transcribed = listen_and_transcribe()

    if transcribed:
        print(f"You said: {transcribed}")

        fixed_sentence = fix_grammar(transcribed)
        print(f"Fixed Text: {fixed_sentence}")
        asyncio.run(stream_voice_clone(fixed_sentence))
    else:
        print("No input found...")
