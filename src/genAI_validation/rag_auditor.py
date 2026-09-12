from __future__ import annotations

import os


class SR117RAGAuditor:
    def __init__(self, pdf_path="data/raw/sr11-7.pdf"):
        self.pdf_path = pdf_path
        self.vectorstore = None
        self.ready = False
        if not os.getenv("OPENAI_API_KEY") or not os.path.exists(pdf_path):
            return
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            docs = PyPDFLoader(pdf_path).load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            self.vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
            self.ready = True
        except Exception:
            self.ready = False

    def audit_model_output(self, draft_report_text: str) -> str:
        if not self.ready:
            return (
                "RAG audit skipped (no SR 11-7 PDF or API key). "
                "Manual gap: this demo does not include independent validation staff, "
                "ongoing monitoring cadence, or challenger sign-off."
            )
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        docs = self.vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(
            "conceptual soundness outcomes analysis monitoring documentation"
        )
        ctx = "\n".join(d.page_content for d in docs)
        prompt = PromptTemplate.from_template(
            "You are checking a demo model card against SR 11-7 excerpts.\n"
            "EXCERPTS:\n{ctx}\n\nDRAFT:\n{draft}\n\n"
            "List only gaps that the excerpts actually require. 3 bullets."
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return (prompt | llm | StrOutputParser()).invoke({"ctx": ctx, "draft": draft_report_text})
