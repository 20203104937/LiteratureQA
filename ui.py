"""Streamlit 页面，使用本机文献问答接口。"""

import os
import requests
import streamlit as st

st.set_page_config(page_title="LiteratureQA 文献问答", layout="wide")
st.title("LiteratureQA 文献问答")
st.caption("从已导入的文献寻找依据，并展示可核对的引用原文。")
endpoint = os.environ.get("LITERATUREQA_BACKEND", "http://127.0.0.1:8000").rstrip("/")
with st.form("question"):
    question = st.text_area("输入问题", placeholder="这些文献提出了哪些检索方法？")
    submit = st.form_submit_button("寻找证据并回答")

if submit:
    if not question.strip():
        st.warning("请先输入问题。")
    else:
        try:
            with st.spinner("正在检索文献和核对证据……"):
                response = requests.post(endpoint + "/ask", json={"question": question}, timeout=1800)
                response.raise_for_status()
                result = response.json()
            st.session_state["result"] = result
        except (requests.RequestException, ValueError):
            st.error("服务暂时无法返回结果，请检查后端状态。")

result = st.session_state.get("result")
if result:
    st.write(result["answer"])
    for reference in result.get("references", []):
        with st.expander(f'[{reference["number"]}] {reference["title"]}'):
            st.write(reference["quote"])
            st.caption(f'来源：{reference["source"]}；全文字符位置：{reference["start"]} 至 {reference["end"]}（不含终点）；阶段：{reference["phase"]}')
    if result.get("errors"):
        st.warning("部分步骤未成功，结果可能遗漏证据。")
    with st.expander("查看问题拆分、筛选规则与调用记录"):
        st.json(result)
