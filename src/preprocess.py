import os
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_PATH = "data/appointments.csv"
OUTPUT_PATH = "data/processed_appointments.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

if not os.path.exists(INPUT_PATH):
    print("Original dataset not found:")
    print(INPUT_PATH)
    exit()

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print("Shape:", df.shape)


# ============================================================
# 3. CONVERT DATE COLUMNS
# ============================================================

df["ScheduledDay"] = pd.to_datetime(
    df["ScheduledDay"],
    errors="coerce"
)

df["AppointmentDay"] = pd.to_datetime(
    df["AppointmentDay"],
    errors="coerce"
)


# ============================================================
# 4. CREATE WAITING DAYS
# ============================================================

df["Waiting_Days"] = (
    df["AppointmentDay"] - df["ScheduledDay"]
).dt.days


negative_waiting = (df["Waiting_Days"] < 0).sum()

print("\nNegative waiting-day records:", negative_waiting)


# Remove invalid waiting periods
df = df[df["Waiting_Days"] >= 0].copy()


# ============================================================
# 5. REMOVE INVALID AGE VALUES
# ============================================================

invalid_age = (df["Age"] < 0).sum()

print("Invalid age records:", invalid_age)

df = df[df["Age"] >= 0].copy()


# ============================================================
# 6. CREATE DATE FEATURES
# ============================================================

df["Appointment_Day"] = df["AppointmentDay"].dt.day_name()

df["Appointment_Month"] = (
    df["AppointmentDay"].dt.month
)

df["Appointment_Hour"] = (
    df["ScheduledDay"].dt.hour
)


# ============================================================
# 7. CREATE AGE GROUP
# ============================================================

def create_age_group(age):

    if age <= 17:
        return "0-17"

    elif age <= 30:
        return "18-30"

    elif age <= 45:
        return "31-45"

    elif age <= 60:
        return "46-60"

    else:
        return "61+"


df["Age_Group"] = df["Age"].apply(create_age_group)


# ============================================================
# 8. CONVERT GENDER
# ============================================================

df["Gender"] = df["Gender"].map({
    "F": 0,
    "M": 1
})


# ============================================================
# 9. CONVERT TARGET
# ============================================================

df["No-show"] = df["No-show"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 10. REMOVE IDENTIFIER COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "PatientId",
        "AppointmentID"
    ],
    errors="ignore"
)


# ============================================================
# 11. REMOVE ORIGINAL DATE COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "ScheduledDay",
        "AppointmentDay"
    ],
    errors="ignore"
)


# ============================================================
# 12. ONE-HOT ENCODE CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "Neighbourhood",
    "Appointment_Day",
    "Age_Group"
]

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)


# ============================================================
# 13. REMOVE MISSING VALUES
# ============================================================

df = df.dropna()


# ============================================================
# 14. SAVE PROCESSED DATA
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("Final shape:", df.shape)

print("\nTarget distribution:")
print(df["No-show"].value_counts())

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nFirst 5 rows:")
print(df.head())