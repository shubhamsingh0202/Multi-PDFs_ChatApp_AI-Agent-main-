# Multi-PDFs 📚 ChatApp AI Agent 🤖 (Groq Edition)

Meet the Multi-PDF Chat AI App! 🚀 Chat seamlessly with multiple PDF documents using LangChain, Groq API (Llama-3.3-70b), and BM25 Retriever, deployed as a sleek Streamlit web application. Get instant, accurate responses powered by ultra-fast Groq inference. 📚💬 Transform your document analysis experience! 🔥✨

## 📝 Description
The Multi-PDF Chat Agent is a Streamlit-based web application designed to facilitate interactive conversations with your documents. The app allows users to upload multiple PDF files, parse and chunk their text, build a high-performance keyword index (BM25), and query them using natural language.

---

## 🎯 How It Works:

The application follows these steps to provide responses to your questions:

1. **PDF Loading**: The app reads uploaded PDF documents and extracts their raw text content.
2. **Text Chunking**: The extracted text is divided into smaller, semantic chunks (Recursive Character Splitting) to fit within LLM context windows.
3. **Retrieval Indexing**: It builds a local **BM25 indexer** over the text chunks. BM25 is an advanced, highly effective keyword-matching algorithm that performs search instantly without needing costly embedding generation or heavy local GPU resources.
4. **Similarity Matching**: When you ask a question, the BM25 indexer retrieves the most relevant text chunks from your documents.
5. **Response Generation**: The retrieved text chunks are sent to the **Groq Llama-3.3-70b** model alongside your question and conversation history to construct a highly context-aware, accurate response.

--- 

## 🎯 Key Features

- **Dynamic API Key Entry**: Input and authenticate your Groq API Key directly through the sidebar UI or pre-fill it via a local `.env` file.
- **Fast Local Indexing**: Powered by BM25, the application indexes documents instantly without waiting for API embedding models.
- **Multi-Document Conversational QA**: Supports queries across multiple documents simultaneously, resolving multi-hop questions.
- **Graceful Error Handling**: Input validation and clear UI warnings for authentication issues or document processing errors.

---

## 🌟 Requirements

The following dependencies are required to run the application:
- **Streamlit**: For building the interactive web user interface.
- **Groq**: The official Python SDK to interact with ultra-fast Groq inference endpoints.
- **LangChain & LangChain-Community**: For document parsing, text splitting, and retriever pipelines.
- **PyPDF2**: For reading and extracting text from PDF documents.
- **rank_bm25**: For the BM25 search algorithm implementation.
- **python-dotenv**: To load environment variables (like API keys) from a local `.env` file.

---

## ▶️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shubhamsingh0202/Multi-PDFs_ChatApp_AI-Agent-main-.git
   ```

2. **Install Required Packages**:
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key (Optional)**:
   You can pre-configure your Groq API key by creating a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
   *Note: If you do not create a `.env` file or leave it blank, you can enter the API key directly in the sidebar of the running app.*

4. **Run the Streamlit Application**:
   ```bash
   streamlit run chatapp.py
   ```

---

## 💡 Usage

1. Open the application in your browser (typically `http://localhost:8501`).
2. If you haven't set up the `.env` file, enter your **Groq API Key** in the sidebar.
3. Upload your PDF documents in the sidebar section.
4. Click on **⚡ Process Documents** to extract and index the text.
5. Once you see the **Ready to Chat** badge, type your question in the text input box at the bottom and press enter!
6. Click **🗑️ Clear Chat** in the sidebar to reset the conversation history.

---

## ©️ License 🪪 

Distributed under the MIT License. See `LICENSE` for more information.

---

#### **If you like this LLM Project, please drop a ⭐ to this repo!**
