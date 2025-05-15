# 📺 Analyzing *Black Clover* with NLP + LLMs

This is an NLP-powered system for analyzing the *Black Clover* anime. It brings together web scraping, theme extraction, character networks, text classification, and an LLM-based character chatbot — all accessible through a Gradio app.

---

## 🧠 What It Does

- 📥 **Scrapes** character dialogue and episode summaries using Scrapy  
- 🧾 **Extracts themes** from episode transcripts using zero-shot classification  
- 👥 **Builds a character network** from named entity recognition and graph analysis  
- 🧪 **Trains a custom text classifier** to label quotes or scenes  
- 🤖 **Simulates conversations** with *Black Clover* characters using an LLM  
- 🖥️ **Runs in a browser** via a Gradio web app

---

## 🗂️ Project Structure

```
.
├── crawler/               # Scrapes Black Clover episode/character data
├── character_network/     # spaCy + NetworkX + PyVis for visual character mapping
├── text_classifier/       # Hugging Face-based custom classifier
├── theme_classifier/      # Zero-shot theme detector
├── character_chat_bot/    # LLM-powered character chatbot
├── gradio_app.py          # Runs the full system in an interactive Gradio UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- **Python**
- **Scrapy** – web scraping  
- **spaCy** – named entity recognition  
- **NetworkX + PyVis** – character graph visualization  
- **Hugging Face Transformers** – zero-shot and fine-tuned models  
- **Gradio** – web interface  
- **PyTorch** – model backend  

---

## 🧬 Model on Hugging Face

This project includes a custom instruct-tuned model published on Hugging Face:

➡️ [**Clover_Llama-3.1-8B-Instruct**](https://huggingface.co/tukyo/Clover_Llama-3.1-8B-Instruct)

The model was designed to simulate characters from *Black Clover* based on structured subtitles and script data. It powers the chatbot and some of the classification features.

---

## 🧪 Example Use Cases

- View a relationship graph of major characters in the show  
- Classify new quotes by tone, character, or plot category  
- Extract themes across arcs (e.g. loyalty, rivalry)  
- Chat with characters like Asta, Yuno, or Noelle using real lines and modeled personas  

---

## 📸 Screenshots

### 🔍 Interface Preview  
A look at the main Gradio web interface that ties all components together.
![Black Clover NLP Interface](https://github.com/user-attachments/assets/eb9492b2-53d2-4f98-9819-428ad5caa2ea)

### 🤖 Chatbot in Action  
Example interaction with the character chatbot built using LLMs and dialogue from the show.
![Black Clover NLP Chatbot Demo](https://github.com/user-attachments/assets/a286ed1f-02b3-4584-a9c7-d5509cc88588)

### 📊 App Walkthrough  
An animated walkthrough showing how the web app processes input and delivers results.
![Black Clover NLP Walkthrough](https://github.com/user-attachments/assets/6e79467c-8f1c-4f14-b8b1-893932d57ade)

### 🎬 Subtitle Input (.ass format)  
This is a snippet of the raw subtitle file used as input for the NLP pipeline. These lines are cleaned and parsed before being passed to the classifier and NER modules.
![Subtitle Sample](https://github.com/user-attachments/assets/95068eb4-dffa-4f2d-afab-56e43b0397ac)

### 🕵️ Named Entity Recognition Output  
A snapshot of the NER results extracted from the subtitle data. This output feeds into the character network builder to visualize relationships and character prominence across episodes.
![NER Output](https://github.com/user-attachments/assets/777daff4-53a0-48a0-965a-05fc249c92b9)

### 🎭 Theme Classification Output  
Example rows from the theme classifier's output. Each episode is analyzed using a zero-shot model to determine the dominant themes, along with confidence scores.

![Theme Classifier Output](https://github.com/user-attachments/assets/6dc9207c-b22b-462b-b73d-fb0e8171978f)
