import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    "你是严谨的文档问答助手。请仅根据下方【上下文】回答用户问题，"
    "并在相关句末标注引用编号，例如[1]、[2]。\n"
    "如果上下文中包含与用户问题相关的具体信息（如姓名、学校、时间等），"
    "即使表述方式不完全相同，也请基于上下文合理推断并回答；"
    "若完全不存在相关信息，请回答：未在文档中找到相关内容。\n\n"
    "【上下文】\n{context}\n\n"
    "【问题】{question}\n\n"
    "【回答】"
)

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

LOADER_MAP = {
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": lambda p: Docx2txtLoader(p),
    ".pdf": lambda p: PyPDFLoader(p),
}

def resolve_path(f):
    return f.name if hasattr(f, "name") else f

def load_documents(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in LOADER_MAP:
        raise ValueError(f"不支持的格式：{ext}，仅支持 .txt/.md/.docx/.pdf")
    return LOADER_MAP[ext](path).load()

def ask(file, question):
    docs = load_documents(resolve_path(file))
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    persist = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

demo = gr.Interface(
    fn=ask,
    inputs=[
        gr.File(label="上传文档", file_types=[".txt", ".md", ".docx", ".pdf"]),
        gr.Textbox(label="你的问题", lines=1)
    ],
    outputs=gr.Textbox(label="回答（带引用）"),
    title="本地文档 RAG 问答助手"
)

if __name__ == "__main__":
    demo.launch()
