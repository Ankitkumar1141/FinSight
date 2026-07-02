import httpx
import streamlit as st

DEFAULT_API_URL = "http://localhost:8008/api/v1"
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".md"]


def get_api_url() -> str:
    api_url = st.sidebar.text_input(
        "FinSight Backend URL",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="Enter the FastAPI backend URL for the FinSight service.",
    )
    st.session_state["api_url"] = api_url.strip().rstrip("/")
    return st.session_state["api_url"]


@st.cache_data(show_spinner=False)
def fetch_health(api_url: str) -> dict:
    with httpx.Client(timeout=20.0) as client:
        response = client.get(f"{api_url}/health")
        response.raise_for_status()
        return response.json()


@st.cache_data(show_spinner=False)
def fetch_documents(api_url: str) -> list:
    with httpx.Client(timeout=20.0) as client:
        response = client.get(f"{api_url}/documents")
        response.raise_for_status()
        return response.json()


def upload_document(api_url: str, uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> dict:
    with httpx.Client(timeout=600.0) as client:  # 10 min — semantic chunking on CPU is slow
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.read(),
                uploaded_file.type or "application/octet-stream",
            )
        }
        response = client.post(f"{api_url}/upload", files=files)
        response.raise_for_status()
        return response.json()


def query_documents(
    api_url: str,
    query_text: str,
    top_k: int,
    filter_doc_type: str,
    filter_year: str,
    stream: bool,
) -> dict:
    payload = {
        "query": query_text,
        "top_k": top_k,
        "filter_doc_type": filter_doc_type or None,
        "filter_year": filter_year or None,
        "stream": stream,
    }
    with httpx.Client(timeout=120.0) as client:
        if stream:
            # FastAPI returns Transfer-Encoding: chunked for streaming responses.
            # Must use httpx's stream() context manager — a plain .post() call
            # cannot consume chunked encoding and raises RemoteProtocolError.
            with client.stream("POST", f"{api_url}/query", json=payload) as response:
                response.raise_for_status()
                answer = "".join(response.iter_text())
            return {"query": query_text, "answer": answer, "sources": []}
        else:
            response = client.post(f"{api_url}/query", json=payload)
            response.raise_for_status()
            return response.json()



def refresh_index_data(api_url: str):
    fetch_health.clear()
    fetch_documents.clear()
    return fetch_health(api_url), fetch_documents(api_url)


def render_source_table(sources: list) -> None:
    if not sources:
        st.info("No source citations returned.")
        return
    st.table(
        [
            {
                "Document": source.get("document", "Unknown"),
                "Page": source.get("page", "N/A"),
                "Type": source.get("doc_type", "unknown"),
                "Year": source.get("year", "unknown"),
                "Company": source.get("company", "unknown"),
                "Score": source.get("relevance_score", 0.0),
            }
            for source in sources
        ]
    )


def main() -> None:
    st.set_page_config(page_title="FinSight UI", page_icon="📊", layout="wide")
    st.title("FinSight")
    st.write(
        "This UI connects to the FinSight FastAPI backend and provides document upload, query, and index inspection capabilities."
    )

    api_url = get_api_url()

    try:
        health = fetch_health(api_url)
        documents = fetch_documents(api_url)
        st.sidebar.success("Backend reachable")
        st.sidebar.write(
            f"**Index:** {health.get('total_chunks', 0)} chunks across {health.get('total_sources', 0)} sources"
        )
    except Exception as exc:
        st.sidebar.error("Backend unavailable")
        st.error(
            "Could not connect to the FinSight backend. Please start `main.py` and ensure the backend is reachable at the configured URL."
        )
        st.exception(exc)
        return

    tab_query, tab_upload, tab_docs = st.tabs(["Ask a Question", "Upload Document", "Index Explorer"])

    with tab_query:
        st.subheader("Query financial documents")
        query_text = st.text_area("Question", value="What did management say about AI investments?", height=120)
        top_k = st.slider("Result chunks to retrieve", min_value=1, max_value=10, value=5)
        filter_doc_type = st.text_input("Optional document type filter", placeholder="annual_report")
        filter_year = st.text_input("Optional year filter", placeholder="2023")
        stream = st.checkbox("Stream response (plain text)", value=False)

        if st.button("Submit query"):
            if not query_text.strip():
                st.warning("Please enter a query before submitting.")
            else:
                with st.spinner("Retrieving and generating answer..."):
                    try:
                        response = query_documents(
                            api_url=api_url,
                            query_text=query_text,
                            top_k=top_k,
                            filter_doc_type=filter_doc_type.strip(),
                            filter_year=filter_year.strip(),
                            stream=stream,
                        )
                        st.markdown("### Answer")
                        st.markdown(response.get("answer", "_No answer returned._"))
                        st.markdown("### Sources")
                        if stream:
                            st.info(
                                "Source citations are not available in streaming mode. "
                                "Uncheck 'Stream response' to see cited sources."
                            )
                        else:
                            render_source_table(response.get("sources", []))
                    except Exception as exc:
                        st.error("Query failed.")
                        st.exception(exc)

    with tab_upload:
        st.subheader("Upload a financial document")
        uploaded_file = st.file_uploader(
            "Choose a file to upload",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        )
        if uploaded_file is not None:
            st.write(f"Selected file: {uploaded_file.name}")
            if st.button("Upload to FinSight"):
                with st.spinner(
                    "Uploading and indexing document — this may take **several minutes** "
                    "for large PDFs (semantic chunking + embedding on CPU). Please wait…"
                ):
                    try:
                        result = upload_document(api_url, uploaded_file)
                        st.success(result.get("message", "Upload completed."))
                        st.write("**Filename:**", result.get("filename"))
                        st.write("**Chunks created:**", result.get("chunks_created"))
                        st.write("**Document type:**", result.get("doc_type"))
                        st.write("**Year:**", result.get("year"))
                        # Auto-refresh so the sidebar count updates immediately
                        health, documents = refresh_index_data(api_url)
                        st.sidebar.write(
                            f"**Index:** {health.get('total_chunks', 0)} chunks across {health.get('total_sources', 0)} sources"
                        )
                        st.info("Index status has been refreshed automatically.")
                    except Exception as exc:
                        st.error("Upload failed.")
                        st.exception(exc)

    with tab_docs:
        st.subheader("Indexed documents and index health")
        if st.button("Refresh index status"):
            with st.spinner("Refreshing index status..."):
                try:
                    health, documents = refresh_index_data(api_url)
                    st.success("Index status refreshed.")
                except Exception as exc:
                    st.error("Failed to refresh index status.")
                    st.exception(exc)

        st.markdown("#### Health")
        st.write(health)

        st.markdown("#### Documents")
        if documents:
            st.table(
                [
                    {
                        "Source": doc.get("source", "unknown"),
                        "Type": doc.get("doc_type", "unknown"),
                        "Year": doc.get("year", "unknown"),
                        "Company": doc.get("company", "unknown"),
                    }
                    for doc in documents
                ]
            )
        else:
            st.info("No documents are indexed yet.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Usage:** Start the FinSight backend with `python main.py` and then run `streamlit run src/ui/app.py`."
    )


if __name__ == "__main__":
    main()
