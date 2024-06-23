import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from .forms import DocumentForm

load_dotenv()

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def index(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_files = request.FILES.getlist('pdf_files')
            user_question = form.cleaned_data['question']

            if pdf_files:
                pdf_paths = [default_storage.save(file.name, file) for file in pdf_files]
                pdf_files = [default_storage.open(path) for path in pdf_paths]

                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                conversation = get_conversation_chain(vectorstore)

                request.session['conversation'] = conversation
                request.session['chat_history'] = []

            if user_question:
                conversation = request.session.get('conversation')
                response = conversation({'question': user_question})
                chat_history = response['chat_history']
                request.session['chat_history'] = chat_history

                return render(request, 'pdfchatapp/index.html', {
                    'form': form,
                    'chat_history': chat_history
                })

    else:
        form = DocumentForm()

    return render(request, 'pdfchatapp/index.html', {'form': form})
