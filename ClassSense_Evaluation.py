import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CLASSENSE - MODEL EVALUATION
# ============================================================

print("========================================")
print("       CLASSENSE MODEL EVALUATION")
print("========================================")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ClassSense_combined_dataset_synthetic.csv"
)

MODEL1_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ClassSense_Model1_Optimized.pkl"
)

MODEL2_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ClassSense_Model2_Optimized.pkl"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA AND MODELS
# ============================================================

df = pd.read_csv(DATA_PATH)

model1 = joblib.load(MODEL1_PATH)
model2 = joblib.load(MODEL2_PATH)

print()
print("Dataset loaded:", len(df), "rows")
print("Model 1 loaded: Student Engagement")
print("Model 2 loaded: Classroom Occupancy")


# ============================================================
# 3. MODEL 1 PREPROCESSING
# ============================================================

def preprocess_model1(data):

    X = pd.DataFrame(index=data.index)

    # Facial Expression
    expressions = [
        "Angry",
        "Happy",
        "Neutral",
        "Sad",
        "Surprised"
    ]

    for value in expressions:

        X[f"Facial_Expression_{value}"] = (
            data["Facial_Expression"] == value
        ).astype(int)


    # Posture
    postures = [
        "Leaning Forward",
        "Slouched",
        "Upright"
    ]

    for value in postures:

        X[f"Posture_{value}"] = (
            data["Posture"] == value
        ).astype(int)


    # Class Subject
    subjects = [
        "History",
        "Literature",
        "Mathematics",
        "Science"
    ]

    for value in subjects:

        X[f"Class_Subject_{value}"] = (
            data["Class_Subject"] == value
        ).astype(int)


    # Numeric features
    numeric_features = [
        "Attendance",
        "Interaction_Level",
        "Movement_m",
        "Heart_Rate",
        "Skin_Temperature",
        "Breathing_Rate",
        "Student_Noise_Level",
        "Student_Lighting",
        "Attention"
    ]

    for feature in numeric_features:

        X[feature] = data[feature].values


    # EXACT FEATURE ORDER USED BY MODEL 1

    feature_order = [

        "Facial_Expression_Angry",
        "Facial_Expression_Happy",
        "Facial_Expression_Neutral",
        "Facial_Expression_Sad",
        "Facial_Expression_Surprised",

        "Posture_Leaning Forward",
        "Posture_Slouched",
        "Posture_Upright",

        "Class_Subject_History",
        "Class_Subject_Literature",
        "Class_Subject_Mathematics",
        "Class_Subject_Science",

        "Attendance",
        "Interaction_Level",
        "Movement_m",
        "Heart_Rate",
        "Skin_Temperature",
        "Breathing_Rate",
        "Student_Noise_Level",
        "Student_Lighting",
        "Attention"
    ]

    return X[feature_order]


# ============================================================
# 4. MODEL 1 TARGET
# ============================================================

def create_engagement_class(score):

    """
    Convert Engagement score into:
    
    0 = Low
    1 = Moderate
    2 = High
    """

    if score < 39.9:

        return 0

    elif score <= 62.4336:

        return 1

    else:

        return 2


if "Engagement" not in df.columns:

    print()
    print("ERROR: Engagement column not found.")
    raise SystemExit


y_engagement = df["Engagement"].apply(
    create_engagement_class
)


# ============================================================
# 5. MODEL 1 EVALUATION
# ============================================================

print()
print("========================================")
print(" MODEL 1: STUDENT ENGAGEMENT")
print("========================================")


X1 = preprocess_model1(df)


X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1,
    y_engagement,
    test_size=0.20,
    random_state=42,
    stratify=y_engagement
)


# Prediction

y1_pred = model1.predict(
    X1_test.to_numpy()
)


# Accuracy

accuracy1 = accuracy_score(
    y1_test,
    y1_pred
)


print()
print("Accuracy:", accuracy1)
print("Accuracy (%):", accuracy1 * 100)


# Classification report

print()
print("Classification Report:")
print()

print(
    classification_report(
        y1_test,
        y1_pred,
        labels=[0, 1, 2],
        target_names=[
            "Low Engagement",
            "Moderate Engagement",
            "High Engagement"
        ],
        zero_division=0
    )
)


# Confusion matrix

cm1 = confusion_matrix(
    y1_test,
    y1_pred,
    labels=[0, 1, 2]
)


print()
print("Confusion Matrix:")
print(cm1)


# ============================================================
# 6. SAVE MODEL 1 CONFUSION MATRIX
# ============================================================

fig1, ax1 = plt.subplots(
    figsize=(7, 6)
)

disp1 = ConfusionMatrixDisplay(
    confusion_matrix=cm1,
    display_labels=[
        "Low",
        "Moderate",
        "High"
    ]
)

disp1.plot(
    ax=ax1,
    values_format="d",
    colorbar=False
)

ax1.set_title(
    "ClassSense Model 1 - Student Engagement"
)

ax1.set_xlabel(
    "Predicted Label"
)

ax1.set_ylabel(
    "True Label"
)

plt.tight_layout()


model1_image = os.path.join(
    RESULTS_DIR,
    "Model1_Confusion_Matrix.png"
)

plt.savefig(
    model1_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig1)


print()
print("Model 1 confusion matrix saved:")
print(model1_image)


# ============================================================
# 7. MODEL 2 TARGET
# ============================================================

print()
print("========================================")
print(" MODEL 2: CLASSROOM OCCUPANCY")
print("========================================")


def create_occupancy_class(value):

    """
    Convert Room_Occupancy_Avg into occupancy classes.

    Class 0 = 0.000000
    Class 1 = 0.518172
    Class 2 = 0.791374
    Class 3 = 1.236662

    Small tolerance is used because CSV floating-point
    values may contain extra decimal digits.
    """

    tolerance = 0.00001

    if abs(value - 0.000000) < tolerance:

        return 0

    elif abs(value - 0.518172) < tolerance:

        return 1

    elif abs(value - 0.791374) < tolerance:

        return 2

    elif abs(value - 1.236662) < tolerance:

        return 3

    else:

        return -1


if "Room_Occupancy_Avg" not in df.columns:

    print()
    print("ERROR: Room_Occupancy_Avg column not found.")
    raise SystemExit


y_occupancy = df["Room_Occupancy_Avg"].apply(
    create_occupancy_class
)


# Check invalid values

invalid_count = (
    y_occupancy == -1
).sum()


if invalid_count > 0:

    print()
    print(
        "ERROR:",
        invalid_count,
        "unexpected occupancy values found."
    )

    print()
    print(
        "Unexpected values:"
    )

    print(
        df.loc[
            y_occupancy == -1,
            "Room_Occupancy_Avg"
        ].unique()
    )

    raise SystemExit


# ============================================================
# 8. MODEL 2 FEATURES
# ============================================================

occupancy_features = [

    "CO2_Avg",
    "CO2_Slope_Avg",
    "Room_Sound_Avg",
    "Room_Light_Avg",
    "Room_Temp_Avg",
    "PIR_S6_Activity",
    "PIR_S7_Activity"

]


X2 = df[occupancy_features]


# ============================================================
# 9. MODEL 2 TEST SPLIT
# ============================================================

X2_train, X2_test, y2_train, y2_test = train_test_split(

    X2,
    y_occupancy,

    test_size=0.20,

    random_state=42,

    stratify=y_occupancy

)


# ============================================================
# 10. MODEL 2 PREDICTION
# ============================================================

y2_pred = model2.predict(
    X2_test
)


# Accuracy

accuracy2 = accuracy_score(
    y2_test,
    y2_pred
)


print()
print("Accuracy:", accuracy2)
print("Accuracy (%):", accuracy2 * 100)


# Classification report

print()
print("Classification Report:")
print()

print(
    classification_report(
        y2_test,
        y2_pred,
        labels=[0, 1, 2, 3],
        target_names=[
            "Class 0",
            "Class 1",
            "Class 2",
            "Class 3"
        ],
        zero_division=0
    )
)


# Confusion matrix

cm2 = confusion_matrix(

    y2_test,
    y2_pred,

    labels=[
        0,
        1,
        2,
        3
    ]

)


print()
print("Confusion Matrix:")
print(cm2)


# ============================================================
# 11. SAVE MODEL 2 CONFUSION MATRIX
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(7, 6)
)

disp2 = ConfusionMatrixDisplay(

    confusion_matrix=cm2,

    display_labels=[
        "Class 0",
        "Class 1",
        "Class 2",
        "Class 3"
    ]

)

disp2.plot(
    ax=ax2,
    values_format="d",
    colorbar=False
)

ax2.set_title(
    "ClassSense Model 2 - Classroom Occupancy"
)

ax2.set_xlabel(
    "Predicted Label"
)

ax2.set_ylabel(
    "True Label"
)

plt.tight_layout()


model2_image = os.path.join(
    RESULTS_DIR,
    "Model2_Confusion_Matrix.png"
)


plt.savefig(
    model2_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig2)


print()
print("Model 2 confusion matrix saved:")
print(model2_image)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("       EVALUATION COMPLETE")
print("========================================")

print()

print(
    f"Model 1 Accuracy: {accuracy1 * 100:.2f}%"
)

print(
    f"Model 2 Accuracy: {accuracy2 * 100:.2f}%"
)

print()
print("Generated files:")

print(
    "1.",
    model1_image
)

print(
    "2.",
    model2_image
)

print()
print("========================================")