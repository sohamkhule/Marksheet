import pandas as pd
import os


def _normalize_columns(cols):
    return (
        pd.Index(cols)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )


def _drop_empty_rows_and_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all", axis=0)  # empty rows
    df = df.dropna(how="all", axis=1)  # empty columns 
    df.index = df.index + 1       # Start index from 1
    df.index.name = "Sr_No"       # Name the index column

    return df


def _find_header_row(df: pd.DataFrame) -> int:
    """
    Try to find the header row that contains both something like 'Roll'
    and something like 'Name'. If not found, fallback to 0.
    We search from TOP and also from BOTTOM just in case.
    """
    n_rows = len(df)

    # Keywords that indicate student identifiers
    id_keywords = ["roll", "id", "reg", "student", "admission", "enroll"]
    name_keywords = ["name", "student"]

    # top-down
    for i in range(n_rows):
        row = df.iloc[i]
        cells = [str(x).strip().lower() for x in row.values if pd.notna(x)]
        if not cells:
            continue

        has_id = any(keyword in c for keyword in id_keywords for c in cells)
        has_name = any(keyword in c for keyword in name_keywords for c in cells)

        if has_id and has_name:
            return i

    # bottom-up (marks table is usually near bottom)
    for i in range(n_rows - 1, max(-1, n_rows - 20), -1):  # Check last 20 rows
        row = df.iloc[i]
        cells = [str(x).strip().lower() for x in row.values if pd.notna(x)]
        if not cells:
            continue

        has_id = any(keyword in c for keyword in id_keywords for c in cells)
        has_name = any(keyword in c for keyword in name_keywords for c in cells)

        if has_id and has_name:
            return i

    # If nothing found, look for row with most non-empty cells (likely header)
    max_filled = 0
    max_row = 0
    for i in range(min(20, n_rows)):  # Check first 20 rows
        row = df.iloc[i]
        filled_count = row.notna().sum()
        if filled_count > max_filled:
            max_filled = filled_count
            max_row = i

    return max_row


def _make_unique_columns(cols):
    """
    Ensure column names are unique and non-empty, to avoid pyarrow / Streamlit issues.
    Even if the header row is not perfect, user can still manually map columns in the UI.
    """
    result = []
    seen = {}

    for idx, c in enumerate(cols):
        if pd.isna(c) or str(c).strip() == "":
            base = f"col_{idx}"
        else:
            base = str(c).strip()

        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            name = base

        result.append(name)

    return result


def _read_pdf_table(file) -> pd.DataFrame:
    """
    Extract tables from PDF using tabula-py.
    Returns the first table found or raises an error.
    """
    try:
        import tabula
    except ImportError:
        raise ImportError(
            "PDF support requires 'tabula-py' and Java. Install with: pip install tabula-py"
        )

    try:
        # Extract all tables from PDF
        tables = tabula.read_pdf(file, pages='all', multiple_tables=True)

        if not tables or len(tables) == 0:
            raise ValueError("No tables found in PDF. Please ensure the PDF contains tabular data.")

        # Return the largest table (most likely to be the marksheet)
        largest_table = max(tables, key=lambda x: x.shape[0] * x.shape[1])
        return largest_table

    except Exception as e:
        raise ValueError(f"Could not extract table from PDF: {str(e)}")


def read_marksheet(file, sheet_name=0) -> pd.DataFrame:
    """
    Read the uploaded Excel/CSV/PDF and return a cleaned DataFrame.

    For complex templates like your school marksheet, we:
    - read with header=None
    - detect header row
    - use that row as header
    - make column names unique
    - normalize column names

    Args:
        file: Uploaded file object
        sheet_name: For Excel files, which sheet to read (default: 0 = first sheet)
    """
    # Determine file type and read accordingly
    file_name = file.name if hasattr(file, 'name') else str(file)
    file_extension = os.path.splitext(file_name)[1].lower()

    # Read the file based on extension
    try:
        if file_extension == '.csv':
            # Read CSV file
            df = pd.read_csv(file, header=None, encoding='utf-8')

        elif file_extension == '.xlsx':
            # Read Excel file with openpyxl engine
            df = pd.read_excel(file, sheet_name=sheet_name, header=None, engine='openpyxl')

        elif file_extension == '.xls':
            # Read old Excel file with xlrd engine
            df = pd.read_excel(file, sheet_name=sheet_name, header=None, engine='xlrd')

        elif file_extension == '.pdf':
            # Read PDF file (requires tabula-py)
            df = _read_pdf_table(file)
            # PDFs already come with some structure, but we'll still process them

        else:
            # If extension is unclear, try multiple approaches
            try:
                # Try CSV with different encodings
                df = pd.read_csv(file, header=None, encoding='utf-8')
            except Exception:
                try:
                    df = pd.read_csv(file, header=None, encoding='latin-1')
                except Exception:
                    try:
                        df = pd.read_excel(file, sheet_name=sheet_name, header=None, engine='openpyxl')
                    except Exception:
                        df = pd.read_excel(file, sheet_name=sheet_name, header=None, engine='xlrd')

    except Exception as e:
        raise ValueError(
            f"Could not read file '{file_name}'. Error: {str(e)}. "
            f"Supported formats: Excel (.xlsx, .xls), CSV (.csv), PDF (.pdf with tables)"
        )

    # Validate we have data
    if df is None or df.empty:
        raise ValueError("The file appears to be empty or could not be read properly.")

    # Remove fully empty rows/columns early so header detection is simpler
    df = _drop_empty_rows_and_cols(df)

    if df.empty:
        raise ValueError("No data found after removing empty rows/columns.")

    # Find and use header row
    header_idx = _find_header_row(df)

    if header_idx >= len(df):
        raise ValueError("Could not find a valid header row in the file.")

    header_row = df.iloc[header_idx]
    data = df.iloc[header_idx + 1 :].reset_index(drop=True)

    if data.empty:
        raise ValueError("No data rows found after header row.")

    # 1) Create unique temporary column names based on header cells
    raw_cols = _make_unique_columns(header_row.values)
    data.columns = raw_cols

    # 2) Drop columns that were empty in the header (these usually became 'col_X')
    #    Only drop columns whose corresponding header cell was actually empty / NaN.
    #    This avoids removing legitimate columns that just happen to be named like 'col_xxx'.
    keep_mask = []
    clean_headers = []
    for orig_header_value, temp_col_name in zip(header_row.values, raw_cols):
        # consider it empty if the original header cell was NaN or an empty string
        if pd.isna(orig_header_value) or str(orig_header_value).strip() == "":
            keep_mask.append(False)
        else:
            keep_mask.append(True)
            clean_headers.append(str(orig_header_value).strip())

    # Apply mask; if mask removes all columns (rare), fall back to keeping all columns
    if any(keep_mask):
        data = data.loc[:, [m for m in keep_mask]]
        # Normalize using original (non-temporary) header text for clarity
        data.columns = _normalize_columns(clean_headers)
    else:
        # If no header cells were non-empty (unexpected), keep the temporary names but normalize them
        data.columns = _normalize_columns(raw_cols)

    return data


def get_excel_sheet_names(file) -> list:
    """
    Get list of sheet names from an Excel file.
    Returns empty list if file is not Excel or has issues.
    """
    try:
        file_name = file.name if hasattr(file, 'name') else str(file)
        file_extension = os.path.splitext(file_name)[1].lower()

        if file_extension == '.xlsx':
            xl_file = pd.ExcelFile(file, engine='openpyxl')
            return xl_file.sheet_names
        elif file_extension == '.xls':
            xl_file = pd.ExcelFile(file, engine='xlrd')
            return xl_file.sheet_names
        else:
            return []
    except Exception:
        return []


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect roll, name, attendance and subject columns.
    Returns a dict with keys: roll, name, attendance, subjects
    (Subjects can always be fixed manually in the UI.)
    """
    cols = list(df.columns)

    # 1) roll/ID column - check multiple keywords
    id_keywords = ["roll", "id", "reg", "student", "admission", "enroll"]
    roll_col = next((c for c in cols if any(keyword in c for keyword in id_keywords)), None)

    # 2) name column
    name_keywords = ["name", "student"]
    name_col = next((c for c in cols if any(keyword in c for keyword in name_keywords)), None)

    # 3) attendance (optional)
    att_keywords = ["att", "present", "absence", "days"]
    att_candidates = [c for c in cols if any(keyword in c for keyword in att_keywords)]
    att_col = att_candidates[0] if att_candidates else None

    # 4) meta columns that are NOT subjects
    meta = {roll_col, name_col, att_col}
    meta = {c for c in meta if c is not None}

    # 5) subject columns:
    #    treat a column as subject if majority of its values are numeric
    #    BUT exclude columns that look like IDs (very long numbers)
    subject_cols = []
    for c in cols:
        if c in meta:
            continue

        col_data = df[c]
        # If duplicated names became multiple columns, ensure 1-D Series
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]

        series = pd.to_numeric(col_data, errors="coerce")
        frac_numeric = series.notna().mean()  # 0..1

        # Skip if not enough numeric values
        if frac_numeric <= 0.5:
            continue

        # Check if values look like IDs (very long numbers, typically > 6 digits)
        numeric_values = series.dropna()
        if len(numeric_values) > 0:
            # Check average number length
            avg_value = numeric_values.mean()
            max_value = numeric_values.max()

            # Skip if numbers are too large (likely IDs, not marks)
            # Typical marks: 0-100, IDs: millions to billions
            if max_value > 1000 or avg_value > 200:
                continue

        # Consider it a subject if it passes all checks
        subject_cols.append(c)

    return {
        "roll": roll_col,
        "name": name_col,
        "attendance": att_col,
        "subjects": subject_cols,
    }
