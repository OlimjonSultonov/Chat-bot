# PDF Chatbot

PDF Chatbot is a Streamlit-based application that enables users to interact with PDF documents through natural language queries. By leveraging the power of OpenAI's GPT models and LangChain, this tool provides meaningful responses by retrieving the most relevant sections of a PDF and generating accurate answers.

## Features
PDF Chatbot allows users to upload PDF files and ask questions about their content. It retrieves relevant sections of the document using LangChain’s advanced text splitting and vector search techniques, then generates human-like responses through OpenAI's GPT models. The tool includes a user-friendly interface for seamless interaction.

## Installation
To use this application, ensure you have Python 3.8 or higher and an OpenAI API Key. Clone the repository with `git clone https://github.com/your-username/pdf-chatbot.git` and navigate to the directory. Create a virtual environment using `python -m venv venv` and activate it with `source venv/bin/activate` (or `venv\Scripts\activate` on Windows). Install dependencies with `pip install -r requirements.txt`. Run the app using `streamlit run app.py`.

## Usage
After launching the app, upload a PDF file via the sidebar, enter your OpenAI API key, and click **Chatbotni Ishga Tushirish**. Use the chat input field to ask questions about the PDF, and receive answers based on its content. The chatbot processes your queries by splitting the PDF content into chunks, embedding them with OpenAI's models, and retrieving the most relevant text for generating responses.

## How It Works
When a PDF is uploaded, it is temporarily stored and processed using LangChain's `RecursiveCharacterTextSplitter` to divide the text into smaller chunks. These chunks are vectorized with OpenAI’s embeddings and stored in a FAISS vector store. Upon receiving a query, the system retrieves relevant chunks and generates responses using GPT models.

## Dependencies
The application requires Python libraries including `streamlit` for the user interface, `openai` for integrating GPT models and embeddings, `langchain` for text splitting and retrieval, and `faiss-cpu` for similarity searches. Install them via the `requirements.txt` file using `pip install -r requirements.txt`.

## Example Use Cases
This application is useful for scenarios such as analyzing research papers, summarizing legal documents, querying financial reports, and exploring e-learning materials interactively. It is ideal for anyone looking to extract insights from complex documents quickly.

## Known Limitations
The chatbot may face challenges with very large PDFs or highly complex queries that exceed the token limit of the GPT model. Additionally, the application relies on a valid OpenAI API key to function.

## Future Improvements
Planned enhancements include support for multiple PDF uploads, local storage for processed files, and integration with newer OpenAI models. Additional features like advanced multilingual support and customizable UI themes are also under consideration.

## Folder Structure
