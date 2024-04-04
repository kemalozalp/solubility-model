import rdkit
import joblib
import boto3
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import DataStructs
import pandas as pd
import numpy as np
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier

logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)  # Set RDLogger Level


def create_ecfp(
    mol: rdkit.Chem.rdchem.Mol, radius: int = 3, nbits: int = 1024
) -> np.ndarray:

    mfbitvector = GetMorganFingerprintAsBitVect(mol, radius, nbits)
    arr = np.zeros((1, 0))
    DataStructs.ConvertToNumpyArray(mfbitvector, arr)

    return arr


# Parameters
cutoff = -3.0

# Load Train Data
data_path = "./data/01_raw/tx2c00379_si_002.xlsx"
train_df = pd.read_excel(data_path, sheet_name="S1. Training Data")

# Create ECFPs
print("Creating ROMols...")
ROMols = [
    MolFromSmiles(smiles) for smiles in tqdm(train_df["Standardized_SMILES"].values)
]
print("Creating ECFPs...")
X_train = [create_ecfp(mol) for mol in tqdm(ROMols)]
y_train = [int(1) if val >= cutoff else int(0) for val in train_df["median_WS"].values]

# Train
clf = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=7)
clf.fit(X_train, y_train)

# Save model
model_path = "./models/solubility_model.joblib"
joblib.dump(clf, model_path)
print(f"Saved model to {model_path}")

# Upload model to AWS S3 Bucket
s3_client = boto3.client("s3")
object_name = "solubility_model.joblib"
bucket_name = "solubilityapp"
s3_client.upload_file(model_path, bucket_name, object_name)
print("Uploaded model to the S3 bucket.")
