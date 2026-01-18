# 📊 School Marksheet → Insights Generator  
A smart analytics application built with **Python, Streamlit, Pandas, and AI (Gemini / HuggingFace / OpenAI fallback)** that converts a school marksheet into meaningful insights for teachers and school management.

---

## 🚀 Project Overview
This project allows schools to upload their **Excel/CSV/PDF** marksheets and automatically generate:

- Student-wise performance
- Subject-wise analytics
- Class averages, weak students, toppers
- Professional AI-generated insights & recommendations  
  (Using Gemini / HuggingFace / Fallback AI if no API available)

It automates manual report making and helps teachers understand class performance more effectively.

---

## ✨ Features

### 🔹 **1. Smart Marksheet Parsing**
- Reads Excel, CSV, and even PDF tables.
- Detects header row automatically.
- Removes empty rows & columns.
- Normalizes inconsistent column names.

### 🔹 **2. Auto Column Detection**
The app intelligently identifies:
- Roll Number column  
- Name column  
- Attendance column  
- All subject columns  

Users can manually adjust if needed (high flexibility).

### 🔹 **3. Analytics Dashboard**
- Total students  
- Class average  
- Weak student count  
- Student-wise table: total marks, average marks, weak/strong indicator  
- Subject-wise averages  
- Bar charts & tabular views  

### 🔹 **4. AI Insights Report**
Uses LLM to generate a readable, teacher-friendly report:
- Overall performance summary  
- Weak subjects & improvement plan  
- Strong subjects & best practices  
- Actionable recommendations  

Supports:
- Google Gemini API  
- HuggingFace models  
- Automatic fallback text if no API is configured  

### 🔹 **5. Clean & Modern UI (Streamlit)**
- Sheet selection for multi-sheet Excel files  
- Step-by-step workflow  
- Real-time preview of data  
- Download/export options (optional)

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Frontend UI | Streamlit |
| Backend Logic | Python |
| Data Processing | Pandas |
| File Parsing | pandas, tabula-py |
| AI Models | Gemini API / HuggingFace Inference API / Fallback |
| Charts | Streamlit built-in charts |

---

## 📁 Folder Structure

├── app.py # Main Streamlit app
├── parser.py # File reading & column detection logic
├── analysis.py # Stats & analytics calculations
├── insights_llm.py # AI insights generation logic
├── requirements.txt # Library dependencies
├── README.md # Project documentation
└── sample_marksheet/ # Example data files (optional) 



📂 Sample Input Data

To make testing easier, sample marksheet files are included in the repository.

Location:

data/


Included file:

marksheet.csv – Example school marksheet containing roll numbers, student names, subject marks, and attendance.

You can directly upload this file in the application to test all features without preparing your own dataset.
---

## 🔧 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/sohamkhule/Marksheet.git
cd Marksheet


3️⃣ Install Dependencies
pip install -r requirements.txt 


4️⃣ Add API Keys (optional but recommended)

Create a .env file:

GEMINI_API_KEY=your_key_here
HF_API_KEY=your_huggingface_key_here
OPENAI_API_KEY=your_openai_key_here  

▶️ Run the App
streamlit run app.py


📘 How It Works
Step 1: Upload File

User uploads Excel/CSV/PDF → file is cleaned & parsed.

Step 2: Column Detection

Automatic detection + option to correct for accuracy.

Step 3: Analysis

Total students

Average marks

Weak & strong performers

Subject averages

Step 4: AI Insights

LLM creates a narrative report using context data.

**DEMO VEDIO LINK** 
https://drive.google.com/file/d/1dUEflQcNEZ-NsvoL2UbRZs8AZfPGfrn1/view?usp=drive_link

🧑‍💻 Author

Soham Khule
BTech Artificial Intelligence & Data Science
AISSMS IOIT, Pune
