import json
import os
import asyncio
from moviepy.editor import AudioFileClip, concatenate_audioclips
from huggingface_hub import InferenceClient
import torch
import edge_tts
import tempfile
import gradio as gr

# Initialize Hugging Face Inference Client
Client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3")
generator = torch.Generator().manual_seed(42)

async def text_to_speech(text, voice, filename):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)

async def generate_conversation(script):
    title = script['title']
    content = script['content']
    
    temp_files = []

    tasks = []
    for key, text in content.items():
        speaker = key.split('_')[0]  # Extract the speaker name
        index = key.split('_')[1]    # Extract the dialogue index
        voice = "en-US-JennyNeural" if speaker == "Alice" else "en-US-GuyNeural"
        
        # Create temporary file for each speaker's dialogue
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_files.append(temp_file.name)
        
        filename = temp_file.name
        tasks.append(text_to_speech(text, voice, filename))
        print(f"Generated audio for {speaker}_{index}: {filename}")

    await asyncio.gather(*tasks)

    # Combine the audio files using moviepy
    audio_clips = [AudioFileClip(temp_file) for temp_file in temp_files]
    combined = concatenate_audioclips(audio_clips)

    # Create temporary file for the combined output
    output_filename = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
    
    # Save the combined file
    combined.write_audiofile(output_filename)
    print(f"Combined audio saved as: {output_filename}")

    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
        print(f"Deleted temporary file: {temp_file}")

    return output_filename

# Function to generate podcast based on user input
async def generate_podcast(topic, seed=42):
    system_instructions = '''[SYSTEM] You are an educational podcast generator. You have to create a podcast between Alice and Bob that gives an overview of the topic given by the user.
  Please provide the script in the following JSON format:
    {
      "title": "[string]",
      "content": {
        "Alice_0": "[string]",
        "BOB_0": "[string]",
        ...
      }
    }
    Be concise.
    '''

    text = f" Topic: {topic} json:"
    formatted_prompt = system_instructions + text
    stream = Client.text_generation(formatted_prompt, max_new_tokens=1024, seed=seed, stream=True, details=True, return_full_text=False)
    
    generated_script = ""
    for response in stream:
        if not response.token.text == "</s>":
            generated_script += response.token.text
    
    print("Generated Script:"+generated_script)

    # Check if the generated_script is empty or not valid JSON
    if not generated_script or not generated_script.strip().startswith('{'):
        raise ValueError("Failed to generate a valid script.")


    script_json = json.loads(generated_script)  # Use the generated script as input
    output_filename = await generate_conversation(script_json)
    print("Output File:"+output_filename)

    # Read the generated audio file
    return output_filename

# Gradio Interface
with gr.Blocks(css="style.css") as demo:    
    with gr.Row():
        input = gr.Textbox(label="User", placeholder="Enter a topic")
        output = gr.Audio(label="AI", type="filepath", interactive=False, autoplay=True, elem_classes="audio")
        
        gr.Interface(
            fn=generate_podcast, 
            inputs=[input],
            outputs=[output],
        )  

if __name__ == "__main__":
    demo.queue(max_size=200).launch()
