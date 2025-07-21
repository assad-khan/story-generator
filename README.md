# 📚 Children's Story Generator with AI 🎨🔊🎬

This Streamlit app generates engaging children's stories complete with **realistic 3D cartoon images**, **AI-generated narration**, and an optional **story video**. Powered by **LLMs**, **TTS**, and **Diffusion models**, the app combines storytelling, illustration, and narration into a seamless interactive experience.

---

## 🚀 Features

* 🎯 **Theme-Based Story Generation** using LLMs (Ollama / GPT / Gemini).
* 📚 Breaks stories into structured, character-consistent scenes.
* 🗣️ Converts each scene into audio using **Text-to-Speech (TTS)**.
* 🖼️ Generates high-quality 3D cartoon-style images for each scene using **Stable Diffusion XL**.
* 🎬 Combines scenes into a video with synchronized image/audio playback.
* ✏️ Interactive **scene editing** interface before exporting the final story.
* 💻 Simple **Streamlit UI** for seamless interaction.

---

## 🧠 Tech Stack

* **Streamlit** – Web interface
* **CrewAI** – Multi-agent orchestration
* **TTS** – Text-to-speech engine (`tacotron2-DDC`)
* **Diffusers** – Image generation via SDXL Turbo
* **Ollama / LLMs** – Story generation (supports `llama3.2`, GPT-4o, Gemini, etc.)
* **MoviePy** – Image/audio stitching into video

---

## 📦 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/childrens-story-generator.git
   cd childrens-story-generator
   ```

2. **Install Dependencies**

   > Use Python 3.9+ and a virtual environment recommended.

   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama (if using locally)**

   ```bash
   ollama run llama3
   ```

4. **Run the App**

   ```bash
   streamlit run app.py
   ```

---

## 🧪 Requirements

* Python 3.9+
* macOS with MPS or GPU-supported machine (or modify for CUDA)
* Local model server (Ollama), or access to GPT/Gemini via API
* FFmpeg (required by `moviepy`)
* At least 8GB RAM recommended

---

## 📁 Project Structure

```
.
├── output/
│   ├── audio/        # Audio narrations per scene
│   └── images/       # Generated scene images
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md
```

---

## ⚙️ Customization

* You can switch LLM providers in `app.py` by editing the `LLM` initialization.
* Modify the `TTS` or `diffusers` model to try different voices or image styles.
* You can extend the `CrewAI` agent setup with your own tools or workflows.

---

## 📸 Demo

![AI Assistant Demo](image_1.png)

---

## 👤 Author

**Asad Khan**
[GitHub](https://github.com/assad-khan) | [Fiverr](https://www.fiverr.com/s/dDB9epg) | [LinkedIn](https://linkedin.com)

---

## 📜 License

```
© 2025 Asad Khan. All rights reserved.

This project is not open for commercial use or redistribution without explicit permission from the author.
