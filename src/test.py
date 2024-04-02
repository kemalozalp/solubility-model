import rdkit
import joblib
import json
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import DataStructs
import pandas as pd
import numpy as np
from rdkit import RDLogger
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

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

# Load Test Data
data_path = "./data/01_raw/tx2c00379_si_002.xlsx"
test_df = pd.read_excel(data_path, sheet_name="S2. Test Data")
print("Creating ROMols...")
ROMols = [
    MolFromSmiles(smiles) for smiles in tqdm(test_df["Standardized_SMILES"].values)
]
print("Creating ECFPs...")
X_test = [create_ecfp(mol) for mol in tqdm(ROMols)]
y_test = [int(1) if val >= cutoff else int(0) for val in test_df["median_WS"].values]

# Test
clf = joblib.load("./models/solubility_model.joblib")
predictions = clf.predict(X_test)

# Evaluate
# acc = accuracy_score(y_test, predictions)
roc = roc_auc_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
mcc = matthews_corrcoef(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
(tn, fp, fn, tp) = cm.ravel()
specificity = tn / (tn + fp)
metrics_dict = {
    "roc": roc,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "specificity": specificity,
    "mcc": mcc,
}
print(pd.DataFrame(metrics_dict, index=[0]))
with open("./results/metrics.json", "w") as out:
    json.dump(metrics_dict, out)
# disp = ConfusionMatrixDisplay(cm)
# disp.plot()
# plt.show()
