import joblib
import os
import rdkit
import base64
import boto3
import numpy as np
import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
from rdkit.Chem import MolFromSmiles, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import RDLogger

# Suppress the warning messages of the RDLogger
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)


# File Loader
def load_file(uploaded_file):
    name, extension = os.path.splitext(uploaded_file.name)
    if extension == ".txt":
        data = pd.read_table(uploaded_file, sep=" ")
    elif extension == ".csv":
        data = pd.read_csv(uploaded_file)
    elif extension == ".xlsx":
        data = pd.read_excel(uploaded_file)
    else:
        print("File format not compatible. Can upload only CSV, XLSX, OR TXT files.")
    return data


# ECFP Creater
def create_ecfp(
    mol: rdkit.Chem.rdchem.Mol, radius: int = 3, nbits: int = 1024
) -> np.ndarray:
    mfbitvector = GetMorganFingerprintAsBitVect(mol, radius, nbits)
    arr = np.zeros((1, 0))
    DataStructs.ConvertToNumpyArray(mfbitvector, arr)
    return arr


# Descriptor Calculater
def calc_desc(data):
    ROMols = [MolFromSmiles(smiles) for smiles in data["SMILES"].values]
    ecfps = [create_ecfp(mol) for mol in ROMols]
    return ecfps


# Predictor
def predict(ecfps):
    model = read_joblib("s3://solubilityapp/solubility_model.joblib")
    prediction = model.predict(ecfps)
    return prediction


# File Downloader
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href


def read_joblib(path):
    """
    Function to load a joblib file from an s3 bucket or local directory.
    Arguments:
    * path: an s3 bucket or local directory path where the file is stored
    Outputs:
    * file: Joblib file loaded
    source: https://stackoverflow.com/questions/62941174/how-to-write-load-machine-learning-model-to-from-s3-bucket-through-joblib
    """

    # Path is an s3 bucket
    if path[:5] == "s3://":
        s3_bucket, s3_key = path.split("/")[2], path.split("/")[3:]
        s3_key = "/".join(s3_key)
        with BytesIO() as f:
            boto3.client("s3").download_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=f)
            f.seek(0)
            file = joblib.load(f)

    # Path is a local directory
    else:
        with open(path, "rb") as f:
            file = joblib.load(f)

    return file


# Logo image
image = Image.open("./docs/logo.png")

st.image(image, use_column_width=True)

# Page title
st.markdown(
    """
# Solubility Classification Model 

This app predicts the solubility of given compounds as a CSV, TXT, or XLSX file.
The solubility model returns '0' or '1' for each compound. '0' -> insoluble, '1' -> soluble.
The solubility cutoff used to train the solubility model is set to '-3' for log10 base.
This means that compounds with solubility <0.001 M or <0.001 mol/L are insoluble to the model.
You can download the predictions as a CSV file.

**Credits**
- App built in `Python` + `Streamlit` by [Kemal Ozalp, Ph.D.](https://kemalozalp.github.io)
- Solubility dataset taken from [Lowe et. al. (2023)](https://pubmed.ncbi.nlm.nih.gov/36877669/)
- Streamlit app inspired tutorial and code by [Data Professor](https://www.youtube.com/watch?v=m0sePkuyTKs&list=PLtqF5YXg7GLlQJUv9XJ3RWdd5VYGwBHrP&index=9)
---
"""
)

# Sidebar
with st.sidebar.header("1. Upload your data"):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input file", type=["csv", "txt", "xlsx"]
    )
    st.sidebar.markdown(
        """
[Example input file](https://github.com/kemalozalp/solubility-model/blob/main/test.csv)
"""
    )

if st.sidebar.button("Predict"):
    data = load_file(uploaded_file)
    # data.to_csv("molecule.smi", sep="\t", header=False, index=False)

    st.header("**Original input data**")
    st.write(data)

    with st.spinner("Calculating descriptors..."):
        ecfps = calc_desc(data)

    # Predict on query compounds
    predictions = predict(ecfps)
    st.header("**Prediction Output**")
    out = data.copy()
    out["Prediction"] = predictions
    st.write(out)
    st.markdown(filedownload(out), unsafe_allow_html=True)
else:
    st.info("Upload input data in the sidebar to start!")
