import os
import torch
import random
import json
import time
from pathlib import Path
from TTS.api import TTS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
# from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
import streamlit as st
from moviepy import AudioFileClip, ImageSequenceClip, concatenate_videoclips


# Create necessary directories
os.makedirs("output/images", exist_ok=True)
os.makedirs("output/audio", exist_ok=True)

# api_key = ''
# os.environ['OPENAI_API_KEY'] = api_key

# LLM Setup
# llm = LLM(model="gpt-4o-mini")
# llm = LLM(model="gemini/gemini-1.5-flash", api_key=api_key)
llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

# Setup device
# If using google colab use this
# device = "cuda" if torch.cuda.is_available() else "cpu"
# If using local system use this
device = "mps"

# Cache the TTS model initialization
@st.cache_resource
def load_tts_model():
    # st.write("Loading TTS model...")
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True).to(device)
    # st.write("TTS model loaded successfully!")
    return tts


@st.cache_resource
def load_sd_pipeline():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to(device)
    return pipe

# Load models (cached)
tts = load_tts_model()
# Load SDXL pipeline and ControlNet
pipe = load_sd_pipeline()

# Update the image generation task
@tool("Generate 3D Image")
def generate_3d_image_tool(scene_text: str, output_path: str) -> str:
    """Generate a realistc 3D Cartoon animation image based on the scene text."""
    image = pipe(prompt= scene_text, num_inference_steps=1, guidance_scale=0.0).images[0]
    image.save(output_path)
    return f"Image saved at {output_path}"


# Streamlit App
st.title("Children's Story Generator")
st.markdown("Welcome to the Children's Story Generator! This app will help you create engaging stories with images, audio, and video.")

# Initialize Session State
if "all_scenes_list" not in st.session_state:
    st.session_state.all_scenes_list = []
if "scene_images" not in st.session_state:
    st.session_state.scene_images = []
if "scene_audios" not in st.session_state:
    st.session_state.scene_audios = []

# Custom CSS for button color
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50; /* Green */
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049; /* Darker green */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables
if "num_scenes" not in st.session_state:
    st.session_state.num_scenes = random.randint(7, 10)

if "story_length" not in st.session_state:
    st.session_state.story_length = random.randint(300, 500)

# Sidebar for user inputs
with st.sidebar:
    # Required Inputs
    theme = st.text_input("🌟 Theme (Required)", "Learning to Share")
    reading_level = st.selectbox("📖 Reading Level (Required)", ["Developing Readers", "Before They Can Read", "Starting to Read", "Confident Readers"])
    moral = st.text_input("💡 Moral Lesson (Required)", "Sharing brings happiness to everyone.")

    # Advanced Settings
    with st.expander("⚙️ Advanced Settings (Optional)"):
        main_character = st.text_input("🎭 Main Character (Optional)", "")
        num_scenes = st.number_input("🎬 Number of Scenes (Optional, Default: 7-10)", min_value=1, max_value=10, value=st.session_state.num_scenes)
        story_length = st.number_input("✍️ Story Length (characters) (Optional, Default: 300-500)", min_value=50, max_value=1000, value=st.session_state.story_length )


@tool("Generate Audio")
def generate_audio(scene_text: str, output_path: str) -> str:
    """Convert scene text to speech and save as an audio file."""
    if not output_path.startswith('output/audio/'):
        output_path = os.path.join('output/audio/', os.path.basename(output_path))
    # sample_audio_path = 'sample1.mp3'
    # tts.tts_to_file(text=scene_text, speaker_wav=sample_audio_path, language="en", file_path=output_path)
    tts.tts_to_file(text=scene_text, file_path=output_path)
    return f"Audio saved at {output_path}"

# Agents
story_generator = Agent(
    role="Story Creator",
    goal="Create an engaging children's story with structured scenes.",
    backstory="A master storyteller weaving fun and educational narratives.",
    verbose=True,
    llm=llm
)

scene_editor = Agent(
    role="Story Editor",
    goal="Break the story into structured scenes.",
    backstory="Ensures smooth transitions and readability.",
    verbose=True,
    llm=llm
)

audio_narrator = Agent(
    role="Voice Actor",
    goal="Convert text into engaging audio narration.",
    backstory="A skilled storyteller bringing words to life.",
    verbose=True,
    llm=llm,
    tools=[generate_audio]
)


image_generator = Agent(
    role="Illustrator",
    goal="Create high-quality realistic 3D cartoon animated images for each scene with character consistency.",
    backstory="An expert AI artist specializing in 3D cartoon animated children’s book illustrations.",
    verbose=True,
    llm=llm,
    tools=[generate_3d_image_tool]
)

# Tasks
story_task = Task(
    description=f"Generate a children's story on '{theme}' with the moral '{moral}'. "
                f"Ensure it's suitable for '{reading_level}'. "
                f"{'Use the main character: ' + main_character if main_character else 'Create an engaging main character with character consistancy.'}",
    expected_output="A structured children's story.",
    agent=story_generator
)

scene_task = Task(
    description="Break the story into {num_scenes} structured scenes with character consistancy. In all secenes describe character. All scenes length are {story_length} characters.",
    expected_output="A JSON list of structured scenes. Each scene is string item of the list.",
    agent=scene_editor
)

generate_audio_task = Task(
    description="Convert scene text '{scene}' into an audio narration using Generate Audio tool.",
    expected_output="A audio files saved in path {audio_path}",
    agent=audio_narrator,
)


image_task = Task(
    description="Generate a realistic 3D cartoon animation image for a scene '{scene}' of the story. Ensure character consistency. "
                "For a scene, construct a valid image prompt and call the 'Generate 3D Image' tool with: scene_text as "
                "a STRING describing the image prompt and output_path as a STRING that is {output_path}.",
    expected_output="A realistic 3D cartoon animation image saved in path 'output/images/' with name",
    agent=image_generator
)

# Button to generate story
with st.sidebar:
    generate_clicked = st.button("Generate Story", key="generate_story_button")

if generate_clicked:
    if not theme or not moral:
        st.error("Theme and Moral Lesson are required.")
        st.stop()

    with st.spinner("Generating story..."):
        crew = Crew(agents=[story_generator, scene_editor], tasks=[story_task, scene_task], process=Process.sequential)
        result = crew.kickoff(inputs={"theme": theme, "reading_level": reading_level, "moral": moral, "num_scenes": num_scenes, "story_length": story_length})
        all_scenes = json.loads(scene_task.output.raw.replace("```json\n", "").replace("\n```", ""))

        st.session_state.all_scenes_list = all_scenes
        st.session_state.scene_images = []
        st.session_state.scene_audios = []

        # Display in column
        cols = st.columns(2)  # Initialize 2 columns

        for i, scene_text in enumerate(all_scenes, start=1):
            image_path = f"output/images/scene_{i}.png"
            audio_path = f"output/audio/scene_{i}.mp3"

            # Generate image
            crew2 = Crew(agents=[image_generator], tasks=[image_task], process=Process.sequential)
            crew2.kickoff(inputs={"output_path": image_path, "scene": scene_text})

            # Generate audio
            crew3 = Crew(agents=[audio_narrator], tasks=[generate_audio_task], process=Process.sequential)
            crew3.kickoff(inputs={"audio_path": audio_path, "scene": scene_text})

            # Save to session state
            st.session_state.scene_images.append(image_path)
            st.session_state.scene_audios.append(audio_path)

            # Display in column (ensure first scene starts on the left)
            col = cols[(i - 1) % 2]  # (i-1) ensures Scene 1 starts on the left column
            with col:
                st.image(image_path, use_container_width=True)
                st.markdown(f"**📖 Scene {i}:** {scene_text}")
                st.audio(audio_path)
                if st.button(f"Edit Scene {i}", key=f"edit_button_{i}"):  # Use unique key here
                    st.session_state.edit_scene_index = i - 1
                    st.rerun()

            # Reset columns after every 2 scenes
            if i % 2 == 0:
                cols = st.columns(2)  # Create new row

# Display scenes in a 2-column grid
def display_scene_grid():
    cols = st.columns(2)
    for idx, (scene_text, img_path, audio_path) in enumerate(zip(
        st.session_state.all_scenes_list,
        st.session_state.scene_images,
        st.session_state.scene_audios
    )):
        col = cols[idx % 2]
        with col:
            st.image(img_path, use_container_width=True)
            st.markdown(f"**📖 Scene {idx + 1}:** {scene_text}")
            st.audio(audio_path)
            if st.button(f"Edit Scene {idx + 1}", key=f"edit_{idx}"):
                st.session_state.edit_scene_index = idx
                st.rerun()

def create_video(image_dir: str, audio_dir: str, output_file: str) -> str:
    """Stitch images and audio together into a video."""
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' is empty or does not exist.")
    if not os.path.exists(audio_dir) or not os.listdir(audio_dir):
        raise FileNotFoundError(f"Audio directory '{audio_dir}' is empty or does not exist.")

    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
    audios = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".mp3")])

    if not images:
        raise FileNotFoundError(f"No PNG images found in directory: {image_dir}")
    if not audios:
        raise FileNotFoundError(f"No MP3 audio files found in directory: {audio_dir}")

    clips = []
    for img, audio in zip(images, audios):
        audio_clip = AudioFileClip(audio)
        img_clip = ImageSequenceClip([img], durations=[audio_clip.duration])
        img_clip = img_clip.set_audio(audio_clip)
        clips.append(img_clip)

    # Use the standalone concatenate_videoclips function
    final_video = concatenate_videoclips(clips)
    final_video.write_videofile(output_file, codec="libx264", fps=24)
    return f"Video saved at {output_file}"




# Function to trigger video creation after all scenes are generated
def generate_video_button():
    if len(st.session_state.scene_images) > 0 and len(st.session_state.scene_audios) > 0:
        # Only show the button once all scenes are generated
        if st.button("Generate Video", key="generate_video_button"):
            with st.spinner("Creating video..."):
                try:
                    # Call the create_video function
                    video_output = "output/story_video.mp4"
                    create_video("output/images", "output/audio", video_output)
                    st.success(f"Video created successfully!")
                    st.video(video_output)
                except Exception as e:
                    st.error(f"Error generating video: {e}")

# Show the scenes in grid layout
if st.session_state.scene_images and not generate_clicked:
    display_scene_grid()

# If all scenes are generated, show the video generation button at the end
if len(st.session_state.scene_images) == len(st.session_state.all_scenes_list):
    generate_video_button()

# Edit Popup
if "edit_scene_index" in st.session_state:
    idx = st.session_state.edit_scene_index
    scene_text = st.session_state.all_scenes_list[idx]
    image_path = st.session_state.scene_images[idx]
    audio_path = st.session_state.scene_audios[idx]

    with st.expander(f"✏️ Edit Scene {idx + 1}", expanded=True):
        st.image(image_path, use_container_width=True)
        new_text = st.text_area("Edit Scene Text", scene_text, height=150)

        if st.button("Save Changes", key=f"save_{idx}"):
            if new_text:
                with st.spinner("Regenerating..."):
                    crew2 = Crew(agents=[image_generator], tasks=[image_task], process=Process.sequential)
                    crew2.kickoff(inputs={"output_path": image_path, "scene": new_text})

                    crew3 = Crew(agents=[audio_narrator], tasks=[generate_audio_task], process=Process.sequential)
                    crew3.kickoff(inputs={"audio_path": audio_path, "scene": new_text})

                    st.session_state.all_scenes_list[idx] = new_text
                    st.success("Scene updated successfully!")
                    del st.session_state.edit_scene_index
                    st.rerun()

        if st.button("Close", key=f"close_{idx}"):
            del st.session_state.edit_scene_index
            st.rerun()

st.markdown("---")
st.markdown("Created with AI ✨")
