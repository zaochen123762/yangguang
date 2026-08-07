import streamlit as st
from zhipuai import ZhipuAI
import PyPDF2
import chromadb
import requests
import os
API_KEY = "db4b7ce1e5384d96b348a877bf0c8f60.T300s1a8zdliQxq2"
client = ZhipuAI(api_key=API_KEY)

def get_embedding(text):
    url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "embedding-2",
        "input": text
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            st.error(f"Embedding API 错误: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"调用 Embedding API 失败: {str(e)}")
        return None
@st.cache_resource
def init_chroma():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="documents")
    return chroma_client, collection
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
def split_text(text, chunk_size=500):
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks
def upload_document(file, collection):
    text = extract_text_from_pdf(file)
    if not text.strip():
        return 0, "PDF 内容为空，可能无法解析"
    chunks = split_text(text)
    doc_id = file.name.replace(".pdf", "")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding is None:
            return 0, "获取 embedding 失败，请检查 API Key"
        chunk_id = f"{doc_id}_{i}"
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[chunk_id],
            metadatas=[{"source": file.name, "chunk": i}]
        )
    return len(chunks), "成功"
def search_documents(query, collection, top_k=3):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    if results['documents'] and len(results['documents'][0]) > 0:
        contexts = []
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            contexts.append(f"【来源：{source}】\n{doc}")
        return "\n\n---\n\n".join(contexts)
    return None
def generate_answer(query, context):
    if context is None:
        return "我在知识库中没有找到与您问题相关的内容。请尝试上传更多相关文档。"
    messages = [
        {"role": "system",
         "content": "你是一个专业的文档问答助手。请基于提供的文档内容回答用户的问题。如果文档内容中没有相关信息，请明确告知用户。"},
        {"role": "user", "content": f"【文档参考内容】\n{context}\n\n【用户问题】\n{query}\n\n请基于上述文档内容回答问题："}
    ]
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages
    )
    return response.choices[0].message.content
st.set_page_config(page_title="智能文档问答系统", layout="wide")
st.title("📚 智能文档问答系统")
st.caption("上传PDF文档，AI将基于文档内容回答你的问题")

chroma_client, collection = init_chroma()
with st.sidebar:
    st.header("📤 上传文档")
    uploaded_files = st.file_uploader("选择PDF文件", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"正在处理 {file.name}..."):
                try:
                    chunk_count, msg = upload_document(file, collection)
                    if chunk_count > 0:
                        st.success(f"✅ {file.name} 已上传（{chunk_count} 个片段）")
                    else:
                        st.error(f"❌ {file.name} 处理失败：{msg}")
                except Exception as e:
                    st.error(f"❌ {file.name} 处理失败：{str(e)}")

    if st.button("🗑️ 清空所有文档"):
        try:
            chroma_client.delete_collection("documents")
        except:
            pass
        st.rerun()
st.header("💬 提问")
try:
    all_data = collection.get()
    if all_data['metadatas']:
        sources = list(set([m['source'] for m in all_data['metadatas']]))
        st.info(f"📄 当前知识库包含 {len(sources)} 份文档：{', '.join(sources)}")
    else:
        st.warning("⚠️ 知识库为空，请先上传PDF文档")
except:
    st.warning("⚠️ 知识库为空，请先上传PDF文档")

query = st.text_input("请输入你的问题：")
if query:
    if st.button("提交问题"):
        with st.spinner("🔍 正在检索相关文档..."):
            context = search_documents(query, collection)
        if context:
            with st.spinner("🤖 AI正在生成回答..."):
                answer = generate_answer(query, context)
                st.markdown("### 📝 回答")
                st.markdown(answer)
                with st.expander("📖 查看参考文档原文"):
                    st.text(context)
        else:
            st.warning("未找到相关文档内容，请尝试上传相关PDF文件")