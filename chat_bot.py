import streamlit as st
st.set_page_config(
    page_title="PDF RAG Chatbot", 
    page_icon="📄", 
    layout="centered"
)

import os
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict
import time

# Custom CSS stillar
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
}

.user-message {
    background-color: #ffffff; /* Oq fon */
    color: #000000; /* Qora matn */
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #4285f4;
}

.assistant-message {
    background-color: #e0e0e0; /* Kulrang fon */
    color: #000000; /* Qora matn */
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #34a853;
}

.stTextInput>div>div>input {
    background-color: #f1f3f4;
    color: #000000; /* Matn qora bo'lsin */
    border: 1px solid #dadce0;
    border-radius: 20px;
    padding: 10px 15px;
}

.stButton>button {
    background-color: #4285f4;
    color: white;
    border-radius: 20px;
    padding: 10px 20px;
    border: none;
}

.stButton>button:hover {
    background-color: #2b65c8; /* Hover effekti uchun quyuq ko'k */
}
</style>
""", unsafe_allow_html=True)


class RAGChatbot:
    def __init__(self, pdf_path: str, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()
        self.pdf_path = pdf_path
        
        # PDF o'qish va bo'laklarga ajratish
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=10
        )
        chunks = text_splitter.split_documents(documents)
        
        # Embedding va vektor do'kon yaratish
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Boshlang'ich messages
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Siz PDF hujjatdagi ma'lumotlar asosida javob beradigan yordam beruvchi AI yordamchisisiz."}
        ]

    def retrieve_context(self, query: str) -> str:
        # Kontekst olish
        docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        return context

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_response(self, query: str) -> str:
        # Kontekst qo'shish
        context = self.retrieve_context(query)
        
        # Promptni shakllantirish
        augmented_query = f"""
        Quyidagi kontekst va savolga asoslanib javob bering:
        
        Kontekst: {context}
        
        Savol: {query}
        
        Agar kontekstda javob yo'q bo'lsa, "Kerakli ma'lumot topilmadi" deb javob bering.
        """
        
        # Yangi message qo'shish
        self.add_message("user", augmented_query)
        
        # OpenAI dan javob olish
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        
        ai_response = response.choices[0].message.content
        self.add_message("assistant", ai_response)
        return ai_response

def type_effect(text):
    """Xabarni bosqichma-bosqich ko'rsatish effekti"""
    placeholder = st.empty()
    full_text = ""
    for word in text.split():
        full_text += word + " "
        placeholder.markdown(full_text)
        time.sleep(0.05)
    return full_text

def main():
    st.title("🤖 PDF Knowledge Chatbot")
    st.markdown("*PDF hujjatlaringiz bilan suhbatlashuvchi AI yordamchisi*")

    # Yan panel
    with st.sidebar:
        st.header("⚙️ Sozlamalar")
        
        # API kalitini kiritish
        api_key = st.text_input(
            "OpenAI API Kaliti", 
            placeholder="API kalitingizni kiriting",
            type="password", 
            help="OpenAI platformasidan olgan API kalitingizni kiriting"
        )
        
        # PDF yuklash
        uploaded_file = st.file_uploader(
            "PDF Hujjatni yuklang", 
            type=["pdf"], 
            help="Kontekst uchun PDF faylni yuklang"
        )
        
        # Model sozlamalari
        model_select = st.selectbox(
            "AI Modeli", 
            ["GPT-4o Mini", "GPT-3.5 Turbo"],
            help="Javob berish uchun model tanlang"
        )
    
    # Agar API key va PDF fayl yuklanmagan bo'lsa
    if not api_key or not uploaded_file:
        st.warning("Iltimos, API kalitingizni kiriting va PDF faylni yuklang.")
        return
    
    # Faylni vaqtincha saqlash
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Chatbot yaratish
    chatbot = RAGChatbot("temp_uploaded.pdf", api_key)
    
    # Suhbat tarozi
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "Salom! PDF hujjatingiz bo'yicha sizga yordam beraman. Nimani bilishni xohlaysiz?"
            }
        ]
    
    # Suhbat konteineri
    chat_container = st.container()
    
    with chat_container:
        # Oldingi xabarlarni ko'rsatish
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Foydalanuvchi kiritmasini qabul qilish
    if prompt := st.chat_input("PDF hujjat haqida so'rang"):
        # Foydalanuvchi xabarini qo'shish
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        
        # AI javobini olish va ko'rsatish
        with st.spinner("Javob tayyorlanmoqda..."):
            response = chatbot.get_response(prompt)
            
            # Type effekti bilan javob ko'rsatish
            displayed_response = type_effect(response)
            
            # Xabarni saqlab qo'yish
            st.session_state.messages.append({"role": "assistant", "content": displayed_response})

if __name__ == "__main__":
    main()