import os
from typing import Optional

import pandas as pd
import streamlit as st
from engine.engine import inference_engine
from engine.utils import load_dataframe
from streamlit_file_browser import st_file_browser


@st.cache_resource
def load_input_data(data_file):
    result = []
    for line in data_file.readlines():
        result.append(line.decode("utf-8").strip().split(","))
    return result

@st.cache_resource
def load_model(cfg:str, ckpt:str, device:str, gpu:Optional[str],\
               cnx_len:int, prd_len:int):
    return inference_engine(cfg, ckpt, device, gpu,
                            cnx_len, prd_len)

load_model_container = st.container(border=True)
with load_model_container:
    st.write("load model")
    cfg_path = ""
    ckpt_path = ""
    st.write("select config file")
    cfg_selected = st_file_browser("baselines", key="cfg", glob_patterns="**/*.py", show_download_file=False, show_preview=False)
    if cfg_selected and cfg_selected["type"] == "SELECT_FILE":
        cfg_path = os.path.join("baselines", cfg_selected["target"]["path"])
    else:
        st.warning("no config selected")

    st.write("select checkpoint")
    ckpt_selected = st_file_browser(".", key="ckpt", glob_patterns="**/*.pt", show_download_file=False, show_preview=False)
    if ckpt_selected and ckpt_selected["type"] == "SELECT_FILE":
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", ckpt_selected["target"]["path"])
    else:
        st.warning("no checkpoint selected")

    device_type = st.selectbox("device type", ["cpu", "gpu", "mlu"])
    gpus = None
    if device_type == "gpu":
        gpus = st.text_input("gpus", "0")

    context_length = int(st.text_input("context length (used for utsf models)", "72"))
    prediction_length = int(st.text_input("prediction length (used for utsf models)", "36"))

    if cfg_path and ckpt_path:
        submitted = st.button("load model")
        if submitted:
            st.write(cfg_path)
            st.session_state["model"] = load_model(cfg_path, ckpt_path, device_type, gpus,\
                                                   context_length, prediction_length)
            st.write("model loaded")

load_data_container = st.container(border=True)
with load_data_container:
    st.write("load input data")
    input_data_file = st.file_uploader("input data file", type=["csv"])
    show_data_df = None
    if input_data_file:
        st.session_state["input_data_list"] = load_input_data(input_data_file)

        st.write("show input data (last 50 rows most)")
        if len(st.session_state["input_data_list"]) > 50:
            show_data = st.session_state["input_data_list"][-50:]
        else:
            show_data = st.session_state["input_data_list"]

        show_data_df = load_dataframe(show_data)
        st.write(show_data_df)
        st.write("data loaded")

    if st.checkbox("show data plot (first 10 columns most)"):
        if show_data_df is not None:
            st.line_chart(show_data_df.iloc[:, :10])
        else:
            st.write("no input data loaded")

inference_container = st.container(border=True)
with inference_container:
    st.write("inference")

    # check model loaded
    if "model" not in st.session_state:
        st.write("model not loaded")
    else:
        if st.button("inference"):
            st.session_state["prediction"] = st.session_state["model"].inference(st.session_state["input_data_list"])
        if "prediction" in st.session_state:
            st.write("inference executed")
            st.write("show prediction data (first 50 rows most)")
            show_pred_data, datetime_data = st.session_state["prediction"]
            if len(show_pred_data) > 50:
                show_pred_data = show_pred_data[:50]
                datetime_data = datetime_data[:50]

            show_pred_data_df = pd.DataFrame(show_pred_data)
            show_pred_data_df.index = datetime_data
            st.write(show_pred_data_df)

            if st.checkbox("show prediction plot (first 10 columns most)"):
                if show_pred_data_df is not None:
                    st.line_chart(show_pred_data_df.iloc[:, :10])
                else:
                    st.write("no prediction data")

            pred_data, datetime_data = st.session_state["prediction"]
            output_pd = pd.DataFrame(pred_data)
            output_pd.index = datetime_data
            if st.download_button("Download CSV", output_pd.to_csv().encode("utf-8"), file_name="prediction.csv", mime="text/csv"):
                st.write("download success")


