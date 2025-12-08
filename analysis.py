# analysis.py
import pandas as pd
import numpy as np


def compute_student_stats(df, cols, pass_mark=35):
    subjects = cols["subjects"]
    roll_col = cols["roll"]
    name_col = cols["name"]

    # Convert subject columns to numeric (fixes the error)
    for subj in subjects:
        df[subj] = pd.to_numeric(df[subj], errors='coerce')
    
    # Check if conversion resulted in all NaN (wrong columns selected)
    if df[subjects].isna().all().all():
        raise ValueError(
            "⚠️ Selected subject columns contain no numeric data. "
            "Please verify you selected the correct columns (should contain marks/scores, not IDs)."
        )

    df["total_marks"] = df[subjects].sum(axis=1)
    df["avg_marks"] = df[subjects].mean(axis=1)

    df["is_weak"] = df[subjects].lt(pass_mark).any(axis=1)

    student_stats = df[[roll_col, name_col, "total_marks", "avg_marks", "is_weak"]]
    student_stats = student_stats.sort_values("avg_marks", ascending=False)

    return student_stats


def compute_subject_stats(df, cols):
    subjects = cols["subjects"]
    
    # Convert subject columns to numeric (fixes the error)
    for subj in subjects:
        df[subj] = pd.to_numeric(df[subj], errors='coerce')
    
    subject_avg = df[subjects].mean().reset_index()
    subject_avg.columns = ["subject", "average"]

    strongest = subject_avg.sort_values("average", ascending=False).head(1)
    weakest   = subject_avg.sort_values("average", ascending=True).head(1)

    return {
        "subject_avg": subject_avg,
        "strongest_subject": strongest,
        "weakest_subject": weakest,
    }


def overall_summary(df, cols):
    subjects = cols["subjects"]
    
    # Convert subject columns to numeric (fixes the error)
    for subj in subjects:
        df[subj] = pd.to_numeric(df[subj], errors='coerce')

    class_avg = df[subjects].values.mean()
    pass_mark = 35

    total_students = len(df)
    weak_students = df[subjects].lt(pass_mark).any(axis=1).sum()

    return {
        "class_avg": round(class_avg, 2),
        "total_students": int(total_students),
        "weak_students": int(weak_students),
        "pass_mark": pass_mark,
    }