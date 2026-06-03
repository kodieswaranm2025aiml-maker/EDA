# Exploratory Data Analysis (EDA) Project
## Employee Performance & Salary Dataset

---

## 📁 Project Structure

```
eda_project/
│
├── data/
│   └── employee_dataset.csv       # Dataset (500 employees, 10 features)
│
├── outputs/
│   ├── fig1_distributions.png     # Histogram distributions
│   ├── fig2_categoricals.png      # Categorical bar charts
│   ├── fig3_correlation_heatmap.png  # Pearson correlation heatmap
│   ├── fig4_salary_analysis.png   # Salary deep-dive
│   ├── fig5_performance_satisfaction.png  # Performance & satisfaction
│   ├── fig6_demographics.png      # Gender & city analysis
│   ├── statistical_summary.csv    # Full descriptive statistics
│   └── department_summary.csv     # Aggregated by department
│
├── report/
│   └── EDA_Report.html            # Full structured HTML report ⭐
│
├── generate_data.py               # Dataset generation script
├── eda_analysis.py                # Main EDA script (all visualizations)
└── README.md                      # This file
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### 2. Generate dataset
```bash
python generate_data.py
```

### 3. Run EDA
```bash
python eda_analysis.py
```

### 4. View Report
Open `report/EDA_Report.html` in any web browser.

---

## 📊 Dataset Description

| Column | Type | Description |
|---|---|---|
| EmployeeID | String | Unique employee identifier |
| Age | Integer | Employee age (18–65) |
| Gender | Categorical | Male / Female |
| Education | Categorical | High School / Bachelor / Master / PhD |
| Department | Categorical | Engineering / Marketing / Sales / HR / Finance |
| City | Categorical | Mumbai / Delhi / Bangalore / Chennai / Hyderabad |
| Experience_Years | Integer | Years of work experience |
| Salary | Float | Annual salary in INR |
| Performance_Score | Integer | 1–5 rating |
| Job_Satisfaction | Float | 1–5 rating |

---

## 🔑 Key Findings

1. **Experience is the strongest predictor of salary** (r = 0.786)
2. **Education moderately impacts salary** (r = 0.402)
3. **Engineering** leads in both average salary (₹80,096) and performance (3.50/5)
4. **Performance and Satisfaction strongly correlate** (r = 0.788)
5. **Minimal gender pay gap** observed in the dataset
6. **Mumbai & Delhi** account for ~50% of the workforce

---

## 🛠️ Tools & Libraries

- **Python 3.x**
- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **matplotlib** — Base plotting
- **seaborn** — Statistical visualizations
- **scipy** — Statistical tests (Pearson correlation)

---

*Internship Project | EDA Report | June 2026*
