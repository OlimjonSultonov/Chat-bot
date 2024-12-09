import streamlit as st
import os
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict

class PDFChatbot:
    def __init__(self, pdf_path: str, api_key: str):
        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()
        
        # PDF o'qish va bo'laklarga ajratish
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Kichikroq chunk size
            chunk_overlap=50  # Kamroq overlap
        )

        chunks = text_splitter.split_documents(documents)
        
        # Embedding va vektor do'kon yaratish
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Retriever (qidiruv parametrlarini o'zgartirdik)
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance
            search_kwargs={
                "k": 3,  # Kam dokumentlar, lekin ko'proq relevantlik
                "fetch_k": 10  # Kengaytirilgan qidiruv
            }
        )
        
        # Yaxshilangan system prompt
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system", 
                "content": """Siz PDF hujjatdagi ma'lumotlarni chuqur tahlil qiladigan va aniq javoblar beradigan yuqori malakali AI yordamchisisiz. 
                Har bir savolga quyidagi qoidalarga amal qiling:
                1. Faqat hujjatdagi ishonchli ma'lumotlarga tayanib javob bering
                2. Agar to'liq javob berib bo'lmasa, mavjud ma'lumotlar asosida eng yaxshi izohni bering
                3. Agar ma'lumot yo'q bo'lsa, aniq va xulosa ravishda "Kerakli ma'lumot topilmadi" deng
                4. Javoblarni qisqa va aniq shakllantiring"""
            }
        ]

    def retrieve_context(self, query: str) -> str:
        try:
            # Savolga oid dokumentlarni topish
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return "Hujjatda tegishli ma'lumot topilmadi."
            
            # Dokumentlarni relevantlik bo'yicha tartiblash
            context = "\n\n".join([
                f"Kontekst {i+1} (Relevantlik: {doc.metadata.get('relevance_score', 'Nomaʼlum')}):\n{doc.page_content}" 
                for i, doc in enumerate(docs)
            ])
            
            return context
        except Exception as e:
            return f"Kontekst olishda xato: {e}"

    def get_response(self, query: str) -> str:
        # Kontekst qo'shish
        context = self.retrieve_context(query)
        
        # Kuchaytirilgan prompt
        augmented_query = f"""Quyidagi kontekst va savolni chuqur tahlil qiling:

Kontekst:
{context}

Savol: {query}

Javob berish qoidalari:
- Faqat mavjud kontekst asosida javob bering
- Agar to'liq javob berib bo'lmasa, mavjud ma'lumotlar bilan cheklanish
- Javobingizni aniq, qisqa va ravshan qiling
"""
        
        # Yangi message qo'shish
        self.messages.append({"role": "user", "content": augmented_query})
        
        # OpenAI dan javob olish
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",  # Katta hajmdagi kontekst uchun model
                temperature=0.2,  # Aniqlikni oshirish
                max_tokens=500,  # Javob uzunligini cheklash
                messages=self.messages
            )
            
            ai_response = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": ai_response})
            return ai_response
        except Exception as e:
            return f"Javob olishda xato: {e}"

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.title("📄 PDF Savol-Javob Chatboti")
    st.markdown("PDF hujjat bo'yicha savollaringizga aniq javoblar olish")

    # Yan panel
    with st.sidebar:
        st.header("🔧 Sozlamalar")
        
        # API kalitini kiritish
        api_key = st.text_input(
            "OpenAI API Kaliti", 
            placeholder="API kalitingizni kiriting",
            type="password"
        )
        
        # PDF yuklash
        uploaded_file = st.file_uploader(
            "PDF Hujjatni yuklang", 
            type=["pdf"]
        )
    
    # Agar API key va PDF fayl yuklanmagan bo'lsa
    if not api_key or not uploaded_file:
        st.warning("Iltimos, API kalitingizni kiriting va PDF faylni yuklang.")
        return
    
    # Faylni vaqtincha saqlash
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Chatbot yaratish
    chatbot = PDFChatbot("temp_uploaded.pdf", api_key)
    
    # Suhbat tarozi
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "PDF hujjat bo'yicha aniq va batafsil savol bering."}
        ]
    
    # Oldingi xabarlarni ko'rsatish
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Foydalanuvchi kiritmasini qabul qilish
    if prompt := st.chat_input("PDF hujjat haqida so'rang"):
        # Foydalanuvchi xabarini qo'shish
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Foydalanuvchi xabarini ko'rsatish
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI javobini olish va ko'rsatish
        with st.chat_message("assistant"):
            with st.spinner("Javob tayyorlanmoqda..."):
                response = chatbot.get_response(prompt)
                st.markdown(response)
        
        # Xabarni saqlab qo'yish
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()