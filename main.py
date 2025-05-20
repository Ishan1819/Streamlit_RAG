# import streamlit as st
# import fitz  # PyMuPDF for PDF text extraction
# import chromadb
# import google.generativeai as genai
# import os
# import datetime

# # Handle PyTorch + Streamlit issue
# os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
# os.environ["STREAMLIT_SERVER_WATCH_DELAY"] = "3"

# # Try importing sentence_transformers
# try:
#     from sentence_transformers import SentenceTransformer
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     HAS_SENTENCE_TRANSFORMERS = True
# except Exception as e:
#     st.error(f"Error importing SentenceTransformer: {str(e)}")
#     HAS_SENTENCE_TRANSFORMERS = False
#     embedding_model = None

# # Initialize ChromaDB collection
# try:
#     chroma_client = chromadb.Client()
#     collection = chroma_client.get_or_create_collection(name="pregnancy_docs")
# except Exception as e:
#     collection = None
#     st.error(f"Error initializing ChromaDB: {str(e)}")

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     try:
#         doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#         return "".join([page.get_text() for page in doc])
#     except Exception as e:
#         st.error(f"Error extracting text from PDF: {str(e)}")
#         return ""

# # Add text to ChromaDB
# def add_text_to_chroma(text, doc_id):
#     try:
#         embedding = embedding_model.encode(text).tolist()
#         existing_ids = set(collection.get()["ids"])
#         if doc_id in existing_ids:
#             collection.delete(ids=[doc_id])
#         collection.add(ids=[doc_id], documents=[text], embeddings=[embedding])
#         return True
#     except Exception as e:
#         st.error(f"Error adding text to ChromaDB: {str(e)}")
#         return False

# # Query ChromaDB
# def query_chroma(query, top_k=3):
#     try:
#         query_embedding = embedding_model.encode(query).tolist()
#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k,
#             include=["documents"]
#         )
#         return results
#     except Exception as e:
#         st.error(f"Error querying ChromaDB: {str(e)}")
#         return {"documents": [["No relevant documents found due to an error."]]}

# # Generate response using Gemini
# def generate_response(query, matched_docs, conversation_history):
#     try:
#         if not matched_docs or not matched_docs.get("documents", [[]])[0]:
#             return "Sorry, I couldn't find relevant information in the uploaded document."

#         matched_texts = "\n\n".join(matched_docs["documents"][0])
#         system_prompt = """
#             You are a helpful assistant specializing in pregnancy and postpartum care.
#             Use bullet points, markdown, and highlight key advice clearly.
#         """
#         history_text = "\n".join(conversation_history[-10:])
#         user_prompt = f"User Query: {query}\n\nExtracted Information:\n{matched_texts}\n\nHistory:\n{history_text}"

#         if not st.session_state.get("api_key_configured", False):
#             return "Please configure your Gemini API key in the sidebar first."

#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content(system_prompt + "\n\n" + user_prompt)
#         return response.text.strip()
#     except Exception as e:
#         st.error(f"Error generating response: {str(e)}")
#         return f"Sorry, I encountered an error: {str(e)}"

# # Generate a unique doc ID
# def generate_doc_id(filename):
#     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#     return f"{filename}_{timestamp}"

# # Main Streamlit app
# def main():
#     st.set_page_config(page_title="Pregnancy Guidance Assistant", page_icon="👶", layout="wide")
#     st.title("Pregnancy Guidance Assistant")
#     st.markdown("Upload pregnancy/maternity PDFs and ask questions.")

#     # Init session state
#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []
#     if "documents_uploaded" not in st.session_state:
#         st.session_state.documents_uploaded = []
#     if "api_key_configured" not in st.session_state:
#         st.session_state.api_key_configured = False

#     # Sidebar
#     with st.sidebar:
#         st.header("Configuration")
#         api_key = st.text_input("Enter your Gemini API key", type="password")
#         if api_key and st.button("Save API Key"):
#             try:
#                 genai.configure(api_key=api_key)
#                 st.session_state.api_key_configured = True
#                 st.success("API Key configured!")
#             except Exception as e:
#                 st.error(f"Error configuring API: {str(e)}")
#                 st.session_state.api_key_configured = False

#         if st.session_state.api_key_configured:
#             st.success("API Key Status: Configured ✓")
#         else:
#             st.error("API Key Status: Not Configured ✗")
#             st.info("Get a Gemini API key at https://ai.google.dev/")

#         st.divider()

#         st.header("Upload PDF")
#         uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
#         if uploaded_file is not None:
#             if not collection:
#                 st.error("Database not initialized.")
#             elif st.button("Process Document"):
#                 with st.spinner("Extracting and indexing document..."):
#                     text = extract_text_from_pdf(uploaded_file)
#                     if text:
#                         doc_id = generate_doc_id(uploaded_file.name)
#                         if add_text_to_chroma(text, doc_id):
#                             st.session_state.documents_uploaded.append(uploaded_file.name)
#                             st.success(f"Uploaded and processed: {uploaded_file.name}")

#         if st.session_state.documents_uploaded:
#             st.subheader("Uploaded Documents")
#             for doc in st.session_state.documents_uploaded:
#                 st.write(f"- {doc}")
#             if st.button("Clear All Documents"):
#                 try:
#                     if collection:
#                         ids = collection.get().get("ids", [])
#                         if ids:
#                             collection.delete(ids=ids)
#                     st.session_state.documents_uploaded = []
#                     st.success("All documents cleared.")
#                 except Exception as e:
#                     st.error(f"Error clearing documents: {str(e)}")

#     # Chat UI
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.header("Ask Questions")
#         with st.container(height=400, border=True):
#             for i, message in enumerate(st.session_state.conversation_history):
#                 if i % 2 == 0:
#                     st.chat_message("user").write(message)
#                 else:
#                     st.chat_message("assistant").markdown(message)

#     with col2:
#         st.header("Instructions")
#         st.markdown("""
#         1. Get a Gemini API key from [Google AI Studio](https://ai.google.dev/)
#         2. Enter the key in the sidebar and save
#         3. Upload pregnancy-related PDFs
#         4. Click "Process Document"
#         5. Ask questions like:
#             - What to eat during pregnancy?
#             - How to prepare for delivery?
#             - What are postpartum symptoms?
#         """)

#         if not st.session_state.documents_uploaded:
#             st.info("No documents uploaded yet.")

#     # Chat input
#     query = st.chat_input("Ask something about pregnancy or postpartum care...")
#     if query:
#         st.chat_message("user").write(query)
#         st.session_state.conversation_history.append(query)

#         if not st.session_state.api_key_configured:
#             response = "Please configure your Gemini API key in the sidebar."
#         elif not st.session_state.documents_uploaded:
#             response = "Please upload a document first."
#         elif not collection:
#             response = "Database not initialized properly."
#         else:
#             with st.spinner("Searching..."):
#                 results = query_chroma(query)
#             with st.spinner("Generating answer..."):
#                 response = generate_response(query, results, st.session_state.conversation_history)

#         st.chat_message("assistant").markdown(response)
#         st.session_state.conversation_history.append(response)

# if __name__ == "__main__":
#     main()



import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import chromadb
import google.generativeai as genai
import os
import datetime

# Handle PyTorch + Streamlit issue
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
os.environ["STREAMLIT_SERVER_WATCH_DELAY"] = "3"

# Try importing sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_SENTENCE_TRANSFORMERS = True
except Exception as e:
    st.error(f"Error importing SentenceTransformer: {str(e)}")
    HAS_SENTENCE_TRANSFORMERS = False
    embedding_model = None

# Initialize ChromaDB collection
try:
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="pregnancy_docs")
except Exception as e:
    collection = None
    st.error(f"Error initializing ChromaDB: {str(e)}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Add text to ChromaDB
def add_text_to_chroma(text, doc_id):
    try:
        embedding = embedding_model.encode(text).tolist()
        existing_ids = set(collection.get()["ids"])
        if doc_id in existing_ids:
            collection.delete(ids=[doc_id])
        collection.add(ids=[doc_id], documents=[text], embeddings=[embedding])
        return True
    except Exception as e:
        st.error(f"Error adding text to ChromaDB: {str(e)}")
        return False

# Query ChromaDB
def query_chroma(query, top_k=1):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"]
        )
        return results
    except Exception as e:
        st.error(f"Error querying ChromaDB: {str(e)}")
        return {"documents": [["No relevant documents found due to an error."]]}

# Generate response using Gemini
def generate_response(query, matched_docs, conversation_history):
    try:
        if not matched_docs or not matched_docs.get("documents", [[]])[0]:
            return "Sorry, I couldn't find relevant information in the uploaded document."

        matched_texts = "\n\n".join(matched_docs["documents"][0])
        system_prompt = """
            You are a helpful assistant specializing in given information by the user.
            Use bullet points, markdown, and highlight key advice clearly.
        """
        history_text = "\n".join(conversation_history[-10:])
        user_prompt = f"User Query: {query}\n\nExtracted Information:\n{matched_texts}\n\nHistory:\n{history_text}"

        if not st.session_state.get("api_key_configured", False):
            return "Please configure your Gemini API key in the sidebar first."

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(system_prompt + "\n\n" + user_prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"

# Generate a unique doc ID
def generate_doc_id(filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{filename}_{timestamp}"

# Main Streamlit app
def main():
    st.set_page_config(page_title="Pregnancy Guidance Assistant", page_icon="👶", layout="wide")
    st.title("Pregnancy Guidance Assistant")
    st.markdown("Upload pregnancy/maternity PDFs and ask questions.")

    # Init session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "documents_uploaded" not in st.session_state:
        st.session_state.documents_uploaded = []
    if "api_key_configured" not in st.session_state:
        st.session_state.api_key_configured = False

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Gemini API key", type="password")
        if api_key and st.button("Save API Key"):
            try:
                genai.configure(api_key=api_key)
                st.session_state.api_key_configured = True
                st.success("API Key configured!")
            except Exception as e:
                st.error(f"Error configuring API: {str(e)}")
                st.session_state.api_key_configured = False

        if st.session_state.api_key_configured:
            st.success("API Key Status: Configured ✓")
        else:
            st.error("API Key Status: Not Configured ✗")
            st.info("Get a Gemini API key at https://ai.google.dev/")

        st.divider()

        st.header("Upload PDF")
        uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            if not collection:
                st.error("Database not initialized.")
            elif st.button("Process Documents"):
                with st.spinner("Extracting and indexing documents..."):
                    for pdf_file in uploaded_files:
                        if pdf_file.name in st.session_state.documents_uploaded:
                            st.info(f"Already processed: {pdf_file.name}")
                            continue
                        text = extract_text_from_pdf(pdf_file)
                        if text:
                            doc_id = generate_doc_id(pdf_file.name)
                            if add_text_to_chroma(text, doc_id):
                                st.session_state.documents_uploaded.append(pdf_file.name)
                                st.success(f"Uploaded and processed: {pdf_file.name}")

        if st.session_state.documents_uploaded:
            st.subheader("Uploaded Documents")
            for doc in st.session_state.documents_uploaded:
                st.write(f"- {doc}")
            if st.button("Clear All Documents"):
                try:
                    if collection:
                        ids = collection.get().get("ids", [])
                        if ids:
                            collection.delete(ids=ids)
                    st.session_state.documents_uploaded = []
                    st.success("All documents cleared.")
                except Exception as e:
                    st.error(f"Error clearing documents: {str(e)}")

    # Chat UI
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Ask Questions")
        with st.container(height=400, border=True):
            for i, message in enumerate(st.session_state.conversation_history):
                if i % 2 == 0:
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").markdown(message)

    with col2:
        st.header("Instructions")
        st.markdown("""
        1. Get a Gemini API key from [Google AI Studio](https://ai.google.dev/)
        2. Enter the key in the sidebar and save
        3. Upload pregnancy-related PDFs
        4. Click "Process Document"
        5. Ask questions like:
            - What to eat during pregnancy?
            - How to prepare for delivery?
            - What are postpartum symptoms?
        """)

        if not st.session_state.documents_uploaded:
            st.info("No documents uploaded yet.")

    # Chat input
    query = st.chat_input("Ask something about pregnancy or postpartum care...")
    if query:
        st.chat_message("user").write(query)
        st.session_state.conversation_history.append(query)

        if not st.session_state.api_key_configured:
            response = "Please configure your Gemini API key in the sidebar."
        elif not st.session_state.documents_uploaded:
            response = "Please upload a document first."
        elif not collection:
            response = "Database not initialized properly."
        else:
            with st.spinner("Searching..."):
                results = query_chroma(query)
            with st.spinner("Generating answer..."):
                response = generate_response(query, results, st.session_state.conversation_history)

        st.chat_message("assistant").markdown(response)
        st.session_state.conversation_history.append(response)

if __name__ == "__main__":
    main()
