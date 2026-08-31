import os
import joblib
import pandas as pd


# ============================================================
# CLASSENSE - MANUAL INPUT ML SYSTEM
# ============================================================

print("========================================")
print("       CLASSENSE ML SYSTEM")
print("========================================")


# ============================================================
# 1. LOAD TRAINED MODELS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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


try:
    model1 = joblib.load(MODEL1_PATH)
    model2 = joblib.load(MODEL2_PATH)

    print("Model 1 loaded: Student Engagement")
    print("Model 2 loaded: Classroom Occupancy")

except Exception as error:
    print("ERROR loading models:")
    print(error)
    raise SystemExit


# ============================================================
# 2. MODEL 1 PREPROCESSING
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

    # Exact training feature order
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
# 3. MODEL 1 - STUDENT ENGAGEMENT
# ============================================================

def predict_engagement(student_data):

    processed_data = preprocess_model1(student_data)

    prediction = model1.predict(
        processed_data.to_numpy()
    )[0]

    labels = {
        0: "Low Engagement",
        1: "Moderate Engagement",
        2: "High Engagement"
    }

    return labels[int(prediction)]


# ============================================================
# 4. MODEL 2 - CLASSROOM OCCUPANCY
# ============================================================

def predict_occupancy(environment_data):

    features = [
        "CO2_Avg",
        "CO2_Slope_Avg",
        "Room_Sound_Avg",
        "Room_Light_Avg",
        "Room_Temp_Avg",
        "PIR_S6_Activity",
        "PIR_S7_Activity"
    ]

    environment_data = environment_data[features]

    prediction = model2.predict(
        environment_data
    )[0]

    occupancy_class = int(prediction)

    # Occupancy index values found in your dataset
    occupancy_index = {
        0: 0.000000,
        1: 0.518172,
        2: 0.791374,
        3: 1.236662
    }

    # General interpretation
    occupancy_level = {
        0: "Empty / Very Low",
        1: "Low-Moderate",
        2: "Moderate-High",
        3: "High"
    }

    return (
        occupancy_class,
        occupancy_index[occupancy_class],
        occupancy_level[occupancy_class]
    )


# ============================================================
# 5. MANUAL STUDENT INPUT
# ============================================================

print()
print("========================================")
print("       STUDENT INPUT")
print("========================================")

print()
print("Facial Expression")
print("1. Angry")
print("2. Happy")
print("3. Neutral")
print("4. Sad")
print("5. Surprised")

expression_choice = input("Enter choice (1-5): ")

expression_map = {
    "1": "Angry",
    "2": "Happy",
    "3": "Neutral",
    "4": "Sad",
    "5": "Surprised"
}

if expression_choice not in expression_map:
    print("Invalid choice. Using Neutral.")
    facial_expression = "Neutral"
else:
    facial_expression = expression_map[expression_choice]


print()
print("Posture")
print("1. Leaning Forward")
print("2. Slouched")
print("3. Upright")

posture_choice = input("Enter choice (1-3): ")

posture_map = {
    "1": "Leaning Forward",
    "2": "Slouched",
    "3": "Upright"
}

if posture_choice not in posture_map:
    print("Invalid choice. Using Upright.")
    posture = "Upright"
else:
    posture = posture_map[posture_choice]


print()
print("Class Subject")
print("1. History")
print("2. Literature")
print("3. Mathematics")
print("4. Science")

subject_choice = input("Enter choice (1-4): ")

subject_map = {
    "1": "History",
    "2": "Literature",
    "3": "Mathematics",
    "4": "Science"
}

if subject_choice not in subject_map:
    print("Invalid choice. Using Mathematics.")
    class_subject = "Mathematics"
else:
    class_subject = subject_map[subject_choice]


print()

attendance = float(
    input("Attendance (0 or 1): ")
)

interaction_level = float(
    input("Interaction Level (0-10): ")
)

movement = float(
    input("Movement (0-5): ")
)

heart_rate = float(
    input("Heart Rate: ")
)

skin_temperature = float(
    input("Skin Temperature (°C): ")
)

breathing_rate = float(
    input("Breathing Rate: ")
)

student_noise = float(
    input("Student Noise Level: ")
)

student_lighting = float(
    input("Student Lighting: ")
)

attention = float(
    input("Attention (0 or 1): ")
)


# ============================================================
# 6. CREATE STUDENT DATA
# ============================================================

student_input = pd.DataFrame([{

    "Facial_Expression": facial_expression,
    "Posture": posture,
    "Class_Subject": class_subject,

    "Attendance": attendance,
    "Interaction_Level": interaction_level,
    "Movement_m": movement,
    "Heart_Rate": heart_rate,
    "Skin_Temperature": skin_temperature,
    "Breathing_Rate": breathing_rate,
    "Student_Noise_Level": student_noise,
    "Student_Lighting": student_lighting,
    "Attention": attention

}])


# ============================================================
# 7. MANUAL CLASSROOM INPUT
# ============================================================

print()
print("========================================")
print("       CLASSROOM INPUT")
print("========================================")

print()

co2 = float(
    input("CO2 Average Level (ppm): ")
)

co2_slope = float(
    input("CO2 Slope Average: ")
)

room_sound = float(
    input("Room Sound Average: ")
)

room_light = float(
    input("Room Light Average: ")
)

room_temp = float(
    input("Room Temperature Average (°C): ")
)

pir_s6 = float(
    input("PIR S6 Activity: ")
)

pir_s7 = float(
    input("PIR S7 Activity: ")
)


# ============================================================
# 8. CREATE ENVIRONMENT DATA
# ============================================================

environment_input = pd.DataFrame([{

    "CO2_Avg": co2,
    "CO2_Slope_Avg": co2_slope,
    "Room_Sound_Avg": room_sound,
    "Room_Light_Avg": room_light,
    "Room_Temp_Avg": room_temp,
    "PIR_S6_Activity": pir_s6,
    "PIR_S7_Activity": pir_s7

}])


# ============================================================
# 9. RUN MODEL 1
# ============================================================

engagement_result = predict_engagement(
    student_input
)


# ============================================================
# 10. RUN MODEL 2
# ============================================================

occupancy_class, occupancy_index, occupancy_level = (
    predict_occupancy(environment_input)
)


# ============================================================
# 11. FINAL CLASSENSE RESULT
# ============================================================

print()
print()
print("========================================")
print("          CLASSENSE RESULT")
print("========================================")

print()

print(
    "Student Engagement       :",
    engagement_result
)

print(
    "Classroom Occupancy Class:",
    occupancy_class
)

print(
    "Occupancy Index          :",
    f"{occupancy_index:.6f}"
)

print(
    "Occupancy Level          :",
    occupancy_level
)

print()

print("========================================")
print("        MODEL PERFORMANCE")
print("========================================")

print()

print("Model 1 Accuracy : 94.83%")
print("Model 2 Accuracy : 100.00%")

print()

print("========================================")
print("       PREDICTION COMPLETE")
print("========================================")