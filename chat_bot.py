import os
import streamlit as st
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class PDFChatbot:
    def __init__(self, pdf, api_key):
        # OpenAI API kalitini o'rnatish
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Vaqtincha fayl yaratish
        with open("temp.pdf", "wb") as f:
            f.write(pdf.getvalue())
        
        # PDF yuklash
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        
        # Matn bo'laklarga bo'lish
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        # Embedding yaratish
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Retriever sozlash
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5
            }
        )
        
        self.client = OpenAI()

    def retrieve_context(self, query):
        docs = self.retriever.get_relevant_documents(query)
        return "\n".join([doc.page_content for doc in docs])

    def generate_response(self, query, context):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "PDF hujjat bo'yicha savolga aniq va qisqa javob bering."
                    },
                    {
                        "role": "user", 
                        "content": f"Kontekst: {context}\n\nSavol: {query}"
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Xatolik: {str(e)}"

def main():
    st.set_page_config(page_title="PDF Chatbot")
    
    st.title("📄 PDF Savollar Chatboti")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Sozlamalar")
        
        # PDF yuklash
        uploaded_pdf = st.file_uploader("PDF Faylni Yuklang", type=['pdf'])
        
        # OpenAI API kaliti
        api_key = st.text_input("OpenAI API Kaliti", type="password")
        
        # Chatbot ishga tushirish
        if st.button("Chatbotni Ishga Tushirish"):
            if uploaded_pdf and api_key:
                try:
                    # Chatbot yaratish
                    chatbot = PDFChatbot(uploaded_pdf, api_key)
                    st.session_state.chatbot = chatbot
                    st.success("Chatbot muvaffaqiyatli ishga tushdi!")
                except Exception as e:
                    st.error(f"Xatolik: {e}")
            else:
                st.warning("PDF fayl va API kalitini kiriting")

    # Chat interfeysi
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Oldingi xabarlarni ko'rsatish
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Foydalanuvchi kiritgan savolni olish
    if prompt := st.chat_input("PDF haqida savol bering"):
        # Xabarni saqlash
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Foydalanuvchi xabarini ko'rsatish
        with st.chat_message("user"):
            st.markdown(prompt)

        # Chatbot mavjud bo'lsa javob berish
        if hasattr(st.session_state, "chatbot"):
            # Kontekst olish
            context = st.session_state.chatbot.retrieve_context(prompt)
            
            # Javob yaratish
            response = st.session_state.chatbot.generate_response(prompt, context)
            
            # Javobni ko'rsatish
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Javobni saqlash
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Avval chatbotni ishga tushiring")

if __name__ == "__main__":
    main()