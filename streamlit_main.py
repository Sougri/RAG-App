#pip install -U langchain-community
#pip install --upgrade langchain langchain-google-genai
#pip install pypdf faiss-cpu #sentence-transformers
#pip install streamlit
#pip install nest_asyncio

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os

# Add custom CSS to hide the GitHub icon
hide_elements = """
<style>
.styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK {
    display: none;
}
</style>
"""
st.markdown(hide_elements, unsafe_allow_html=True)

# Print current directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Set the API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

if "GOOGLE_API_KEY" not in os.environ:
    print("No connection with the server")
else:
    print("connected with the server")

# --- Configuration du Prompt et du LLM (peut être mis dans une fonction pour la propreté) ---

prompt_template_francais = """
Vous êtes un assistant juridique expert. Votre rôle est d'expliquer un sujet juridique complexe à une personne qui n'a aucune connaissance dans ce domaine.
Votre objectif est de fournir une réponse simple, claire et instructive en vous basant *uniquement* sur le contexte fourni et l'historique de la conversation. Ne générez pas d'informations qui ne sont pas dans le contexte.

Si vous ne connaissez pas la réponse, dites simplement que vous ne l'avez pas trouvée dans le document. N'essayez pas d'inventer une réponse.

CONTEXTE :
{context}

Historique de la conversation :
{chat_history}

Question :
{question}

Réponse simple et claire :
"""
CUSTOM_PROMPT = PromptTemplate(
    template=prompt_template_francais, input_variables=["chat_history", "question"]
)

# Fonction pour initialiser la chaîne de conversation (pour éviter de la recréer à chaque interaction)
@st.cache_resource
def load_base_dependencies():
    # Charger et découper le document
    path = "./Database" #"/workspaces/RAG-App/Database"
    # Load the legal document
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Embeddings et Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    
    return retriever

def load_chain_with_session_history():
    """
    Loads the ConversationalRetrievalChain with a unique memory for each session.
    """
    retriever = load_base_dependencies()
    # LLM
    llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-2.0-flash")
    
    # Create a unique memory for each session if it doesn't exist
    if 'memory' not in st.session_state:
        print(f"--- Creating new memory for session {st.session_state.session_id} ---")
        st.session_state.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'  # Important for the chain to know where the answer is stored
        )
    
    # The conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
    )
    return chain

# --- Interface utilisateur Streamlit ---

st.set_page_config(page_title="Assistant Juridique", page_icon="⚖️")
st.title("⚖️ Assistant Juridique Personnalisé")
st.markdown("Posez vos questions sur le document juridique chargé, et je vous fournirai une réponse simple et claire.")

# Charger la chaîne conversationnelle
chain = chain = load_chain_with_session_history()

# Initialiser l'historique du chat dans l'état de la session Streamlit
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Afficher les messages précédents
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Gérer la nouvelle entrée de l'utilisateur
if user_prompt := st.chat_input("Posez votre question ici..."):
    # Ajouter le message de l'utilisateur à l'historique et l'afficher
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Obtenir la réponse de l'assistant
    with st.spinner("L'assistant réfléchit..."):
        result = chain({"question": user_prompt})
        response = result["answer"]

        # Afficher la réponse de l'assistant et l'ajouter à l'historique
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
