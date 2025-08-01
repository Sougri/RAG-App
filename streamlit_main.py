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

# --- Configuration de la page et des styles ---
st.set_page_config(page_title="Assistant Juridique", page_icon="⚖️")

# CSS personnalisé pour masquer les badges Streamlit
hide_elements = """
<style>
.styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK {
    display: none;
}
</style>
"""
st.markdown(hide_elements, unsafe_allow_html=True)

# --- Connexion à l'API Google ---
# Il est recommandé de gérer les secrets de cette manière pour la sécurité
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- Définition du Prompt Personnalisé ---
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
    template=prompt_template_francais, input_variables=["chat_history", "context", "question"]
)

# --- Fonctions de chargement (Mise en cache pour la performance) ---

@st.cache_resource
def load_and_process_documents():
    """
    Charge les PDF, les découpe et crée une base de données vectorielle (retriever).
    Cette fonction est mise en cache (@st.cache_resource) car elle est coûteuse et
    son résultat est identique pour toutes les sessions. Elle n'est exécutée qu'une seule fois.
    """
    print("--- Exécution de load_and_process_documents (ne devrait apparaître qu'une fois) ---")
    path = "./Database"
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store.as_retriever()

def get_conversational_chain():
    """
    Crée et configure la chaîne de conversation.
    Cette fonction s'appuie sur st.session_state pour la mémoire,
    garantissant une chaîne unique par session utilisateur.
    """
    retriever = load_and_process_documents()
    llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-2.0-flash")
    
    # POINT CLÉ N°1 : Initialisation de la mémoire propre à la session
    # Ce bloc de code vérifie si 'memory' existe DANS LA SESSION ACTUELLE.
    # S'il n'existe pas (nouvel onglet/nouvel utilisateur), il en crée un.
    # Sinon, il réutilise la mémoire existante pour cet onglet spécifique.
    if 'memory' not in st.session_state:
        print(f"--- Création d'une nouvelle mémoire pour la session ---")
        st.session_state.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
    
    # La chaîne est créée en utilisant la mémoire de la session en cours.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=False # Optionnel : masquer les documents sources dans le résultat
    )
    return chain

# --- Interface utilisateur Streamlit ---

st.title("⚖️ Assistant Juridique Personnalisé")
st.markdown("Posez vos questions sur le document juridique chargé, et je vous fournirai une réponse simple et claire.")

# Charger la chaîne conversationnelle pour la session actuelle
chain = get_conversational_chain()

# POINT CLÉ N°2 : Initialisation de l'historique des messages propre à la session
# De même que pour la mémoire, 'messages' est créé une fois par session
# pour stocker et afficher l'historique du chat.
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Afficher les messages de l'historique de la session en cours
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Gérer la nouvelle entrée de l'utilisateur
if user_prompt := st.chat_input("Posez votre question ici..."):
    # Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Obtenir la réponse de l'assistant
    with st.spinner("L'assistant réfléchit..."):
        result = chain({"question": user_prompt})
        response = result["answer"]

        # Ajouter et afficher la réponse de l'assistant
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
