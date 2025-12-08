import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from parser import read_marksheet, detect_columns
from analysis import compute_student_stats, compute_subject_stats, overall_summary
from insights_llm import build_context, generate_ai_insights

# Load .env (for HF API key check)
load_dotenv()
HF_KEY = os.getenv("HF_API_KEY")

# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="School Marksheet → Insights Generator",
    layout="wide",
)

# ---------- Helper ----------
def _safe_index(columns, col_name, default=0):
    columns = list(columns)
    if col_name is not None and col_name in columns:
        return columns.index(col_name)
    return default


# ---------- UI ----------
st.title("📊 School Marksheet → Insights Generator")

st.write(
    """
Upload a marksheet (Excel/CSV), confirm which columns are roll no, name and subjects,
and get analytics + an AI-style insights report.
"""
)

if not HF_KEY:
    st.warning(
        "Hugging Face API key not found. AI insights will fall back to the offline report. "
        "To enable real AI insights, add HF_API_KEY in your .env (Settings → Access Tokens → Read)."
    )

uploaded_file = st.file_uploader(
    "Upload marksheet (Excel/CSV/PDF)",
    type=["xlsx", "xls", "csv", "pdf"],
    help="Supported formats: Excel (.xlsx, .xls), CSV (.csv), PDF (.pdf with tables)"
)

if uploaded_file is None:
    st.info("⬆️ Please upload a marksheet file to begin.")
else:
    # Check if it's an Excel file with multiple sheets
    sheet_names = []
    selected_sheet = 0
    
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension in ['.xlsx', '.xls']:
        from parser import get_excel_sheet_names
        sheet_names = get_excel_sheet_names(uploaded_file)
        
        if len(sheet_names) > 1:
            st.info(f"📄 This Excel file has {len(sheet_names)} sheets. Please select which one contains the marksheet.")
            selected_sheet_name = st.selectbox(
                "Select Sheet",
                options=sheet_names,
                index=0
            )
            selected_sheet = sheet_names.index(selected_sheet_name)
    
    # ---- 1. Read and show raw data ----
    with st.spinner("Reading marksheet..."):
        try:
            df = read_marksheet(uploaded_file, sheet_name=selected_sheet)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            if "tabula" in str(e).lower() and file_extension == '.pdf':
                st.info("💡 To read PDF files, install: `pip install tabula-py` (requires Java)")
            st.stop()

    df.index = df.index + 1
    df.index.name = "Sr_No"
    st.subheader("Step 1 – Preview of detected data")
    st.dataframe(df.head())

    # ---- 2. Auto column detection ----
    auto_cols = detect_columns(df)

    st.subheader("Step 2 – Column detection")
    st.write("Auto-detected columns (you can change them below):")
    st.json(auto_cols)

    columns_list = df.columns.tolist()

    # ---- 3. Manual override UI ----
    st.subheader("Step 3 – Confirm / adjust column mapping")

    # Roll No column
    roll_col = st.selectbox(
        "Select *Roll No* column",
        options=columns_list,
        index=_safe_index(columns_list, auto_cols.get("roll")),
    )

    # Name column
    name_col = st.selectbox(
        "Select *Name* column",
        options=columns_list,
        index=_safe_index(columns_list, auto_cols.get("name")),
    )

    # Attendance (optional)
    att_auto = auto_cols.get("attendance")
    att_options = ["(none)"] + columns_list
    if att_auto is not None and att_auto in columns_list:
        att_index = columns_list.index(att_auto) + 1
    else:
        att_index = 0

    att_choice = st.selectbox(
        "Select *Attendance* column (optional)",
        options=att_options,
        index=att_index,
    )
    attendance_col = None if att_choice == "(none)" else att_choice

    # Subject columns (marks)
    # Get auto-detected subjects
    auto_subjects = auto_cols.get("subjects", [])
    
    # Filter options: exclude roll, name, and attendance columns
    excluded_cols = {roll_col, name_col}
    if attendance_col is not None:
        excluded_cols.add(attendance_col)
    
    available_subject_options = [c for c in columns_list if c not in excluded_cols]
    
    # Filter default subjects to only include those in available options
    default_subjects = [c for c in auto_subjects if c in available_subject_options]

    subject_cols = st.multiselect(
        "Select *Subject* columns (marks columns)",
        options=available_subject_options,
        default=default_subjects,
        help="Choose all columns that contain subject marks (Sub-1, Sub-2, Maths, Science, etc.).",
    )

    if len(subject_cols) == 0:
        st.warning("Please select at least one subject column above to continue.")
        st.stop()
    
    # Validate that selected columns contain reasonable marks data
    st.info("💡 **Tip:** Subject columns should contain marks/scores (typically 0-100). "
            "If you see very large numbers (like IDs), please reselect the columns.")
    
    # Show a preview of selected subject columns
    with st.expander("📊 Preview of selected subject data"):
        preview_data = df[subject_cols].head(10)
        # Try to convert to numeric to show what will be analyzed
        preview_numeric = preview_data.copy()
        for col in subject_cols:
            preview_numeric[col] = pd.to_numeric(preview_data[col], errors='coerce')
        st.dataframe(preview_numeric)
        
        # Show basic stats to help user verify
        st.write("**Quick Statistics:**")
        for col in subject_cols[:3]:  # Show first 3 subjects
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() > 0:
                st.write(f"- {col}: Min={numeric_col.min():.1f}, Max={numeric_col.max():.1f}, Avg={numeric_col.mean():.1f}")
            else:
                st.warning(f"⚠️ Column '{col}' has no numeric values!")


    # Build final columns mapping
    cols = {
        "roll": roll_col,
        "name": name_col,
        "attendance": attendance_col,
        "subjects": subject_cols,
    }

    # ---- 4. Run analysis ----
    st.subheader("Step 4 – Class analysis")

    with st.spinner("Computing statistics..."):
        try:
            student_stats = compute_student_stats(df.copy(), cols)
            subject_stats = compute_subject_stats(df.copy(), cols)
            summary = overall_summary(df.copy(), cols)
        except ValueError as e:
            st.error(f"❌ Analysis Error: {str(e)}")
            st.warning("Please go back to Step 3 and verify that you selected the correct subject columns. "
                      "Subject columns should contain numeric marks/scores, not IDs or text.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error during analysis: {str(e)}")
            st.stop()

    # Top summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Students", summary["total_students"])
    m2.metric("Class Average", summary["class_avg"])
    m3.metric("Weak Students", summary["weak_students"])

    # Student performance table
    st.markdown("### Student Performance")
    st.dataframe(student_stats)

    # Subject averages + chart
    st.markdown("### Subject-wise Averages")
    st.dataframe(subject_stats["subject_avg"])

    st.markdown("Average marks per subject (bar chart):")
    chart_data = subject_stats["subject_avg"].set_index("subject")["average"]
    st.bar_chart(chart_data)

    # ---- 5. AI Insights ----
    st.subheader("Step 5 – AI Insights Report")

    # Disable the button if no HF key and you want to prevent API attempts
    generate_disabled = False
    # If you prefer to prevent API calls without key, uncomment:
    # generate_disabled = not bool(HF_KEY)

    if st.button("Generate AI Insights", disabled=generate_disabled):
        with st.spinner("Generating AI insights..."):
            context_text = build_context(
                summary,
                subject_stats,
                top_students=student_stats,
                weak_students=student_stats[student_stats["is_weak"]],
            )

            # NOTE: generate_ai_insights expects (llm_client, context_text)
            try:
                # Try new signature (context_text only)
                insights_res = generate_ai_insights(None, context_text)
            except TypeError:
                # Fall back if signature changed
                insights_res = generate_ai_insights(context_text)

            # If HF client returned an object, convert to string safely
            if isinstance(insights_res, dict):
                # some HF responses may be dict-like; try to extract text keys
                insights_text = insights_res.get("generated_text") or str(insights_res)
            else:
                insights_text = str(insights_res)

        st.markdown("#### Generated Insights")
        st.write(insights_text)
    else:
        st.info("Click **Generate AI Insights** to create a narrative report.")