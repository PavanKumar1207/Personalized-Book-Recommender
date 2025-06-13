# 📚 Personalized Book Recommender

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/downloads/)
[![Built with Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange?logo=gradio)](https://gradio.app/)
[![Powered by LangChain](https://img.shields.io/badge/Powered%20by-LangChain-blueviolet)](https://www.langchain.com/)


[![Embeddings by Google Generative AI](https://img.shields.io/badge/Embeddings-Google%20Generative%20AI-lightgrey)](https://ai.google/discover/generative-ai/)
[![Vector DB: Astra DB](https://img.shields.io/badge/Vector%20Store-AstraDB-lightgrey)](https://www.datastax.com/astra)
[![Models from Hugging Face](https://img.shields.io/badge/Models-HuggingFace-yellow)](https://huggingface.co/)

A smart and modern book recommendation system that helps you find your next great read using your query, preferred genre, and emotional tone.  
It combines advanced natural language processing, semantic vector search, and an intuitive Gradio interface for a personalized discovery experience.

---

## 🚀 Features

- 🔎 **Semantic Search** — Understands your reading mood using AI embeddings.
- 🎭 **Mood-Based Filtering** — Choose books based on emotional tones like *Happy*, *Suspenseful*, or *Sad*.
- 📚 **Genre-Based Filtering** — Filter books into simplified categories like *Fiction* and *Non-Fiction*.
- 🖼️ **Visual Dashboard** — Clean and intuitive UI built with Gradio and styled with custom CSS.
- 🧠 **Smart Embeddings** — Powered by Google Generative AI and stored in Astra DB Vector Store.

---

## 🧠 Tech Stack

| Purpose                 | Tools & Libraries |
|------------------------|-------------------|
| Programming Language   | Python            |
| Dashboard UI           | Gradio, Custom CSS |
| NLP Embeddings         | Google Generative AI Embeddings |
| Vector Store           | AstraDB Vector Store via LangChain |
| Sentiment & Emotion    | HuggingFace Transformers |
| Data Processing        | Pandas, NumPy     |
| EDA & Visualization    | Seaborn, Matplotlib |

---

## 🗂️ Project Structure
```
book_recommender/
├── gradio_dashboard.py # Main file to launch Gradio interface
├── vector_search.py # Embedding generation & vector DB integration
├── sentiment_classification.py # Emotion detection from book descriptions
├── text_classification.py # Zero-shot genre classification
├── data_exploration.py # EDA and data visualization
├── create_db.py # feeding the data to the astradb
├── data/
│ ├── books_cleaned.csv
│ ├── books_with_classification.csv
│ └── books_with_emotions.csv
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>```
2. **Create Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate ```
3. **Install Requirements**
    ```bash
    pip install -r requirements.txt```
4. **Set Up .env File**
   Create a .env file in the root directory and add:
   ```bash
   ASTRA_API_ENDPOINT=your_astra_api_endpoint
   ASTRA_DB_TOKEN=your_astra_db_token
   GOOGLE_API_KEY=your_google_api_key```
5. **Launch the Gradio Dashboard**
   ```bash
   python book_recommender/gradio_dashboard.py```
---

## ✅ Example Use

Describe a book you’d like:

`“A suspenseful detective story with emotional depth.”`

**Filter by:**  
- Category: `Fiction`  
- Mood: `Suspenseful`  

And you’ll get personalized recommendations!

---

## 📌 Future Enhancements

- ✨ Add user login and search history  
- 🌍 Host on Hugging Face or Render  
- 📖 Enable user-submitted reviews  
- 📈 Improve emotion accuracy with fine-tuned models  

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

## 🙌 Acknowledgments

- [LangChain](https://www.langchain.com/)
- [Gradio](https://gradio.app/)
- [Hugging Face Transformers](https://huggingface.co/)
- [Google Generative AI](https://ai.google/discover/generative-ai/)
- [Astra DB](https://www.datastax.com/astra)

---

**Made with ❤️ by Pavan :) | ✨ Happy Reading!**
   
   
