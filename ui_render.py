# ui_render.py

from typing import List, Dict, Any
import streamlit as st


def render_chunks(chunks: List[Any], title: str = "Chunks"):
    """
    Render retrieved document or memory chunks in a clean, readable UI.
    Supports:
      - dict chunks with keys like {text, metadata}
      - plain strings
    """
    st.markdown(f"### {title}")

    if not chunks:
        st.info("No chunks to display.")
        return

    for i, ch in enumerate(chunks, start=1):
        with st.expander(f"{title} #{i}", expanded=False):
            if isinstance(ch, dict):
                text = ch.get("text", "")
                meta = ch.get("metadata", {}) or {}

                if text:
                    st.write(text)

                if meta:
                    st.markdown("**Metadata**")
                    st.json(meta)

            else:
                st.write(str(ch))


def render_eval_table(record: Dict[str, Any]):
    """
    Render evaluation results returned by evaluator.evaluate_item().
    Gracefully handles missing keys.
    """
    if not record:
        st.info("No evaluation data to display.")
        return

    st.markdown("### Evaluation Summary")

    # Common scalar fields
    scalar_fields = {
        "accuracy": "Accuracy",
        "relevance": "Relevance",
        "faithfulness": "Faithfulness",
        "completeness": "Completeness",
        "overall": "Overall Score",
    }

    rows = []
    for k, label in scalar_fields.items():
        if k in record:
            rows.append({"Metric": label, "Score": record[k]})

    if rows:
        st.table(rows)

    # Free-form feedback/comments
    for key in ("comments", "notes", "feedback", "explanation"):
        if key in record and record[key]:
            st.markdown(f"**{key.capitalize()}**")
            st.write(record[key])

    # Context / meta
    if "meta" in record:
        with st.expander("Eval Metadata", expanded=False):
            st.json(record["meta"])

    if "sources" in record:
        with st.expander("Sources", expanded=False):
            st.json(record["sources"])
