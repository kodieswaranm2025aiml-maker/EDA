import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

age = np.random.normal(35, 10, n).clip(18, 65).astype(int)
experience = (age - 18 + np.random.normal(0, 2, n)).clip(0, 40).astype(int)
education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.2, 0.45, 0.25, 0.10])
edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
edu_num = np.array([edu_map[e] for e in education])

department = np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], n,
                               p=[0.30, 0.20, 0.20, 0.15, 0.15])

base_salary = 30000
salary = (base_salary
          + experience * 1500
          + edu_num * 8000
          + np.random.normal(0, 5000, n)
          + (department == 'Engineering') * 15000
          + (department == 'Finance') * 8000
          - (department == 'HR') * 2000)
salary = salary.clip(25000, 150000).astype(int)

performance = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.35, 0.30, 0.15])
satisfaction = np.clip(performance + np.random.randint(-1, 2, n), 1, 5)
gender = np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45])
city = np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad'], n,
                        p=[0.25, 0.25, 0.20, 0.15, 0.15])

# Introduce some missing values
salary_col = salary.astype(float)
salary_col[np.random.choice(n, 15, replace=False)] = np.nan
satisfaction_col = satisfaction.astype(float)
satisfaction_col[np.random.choice(n, 10, replace=False)] = np.nan

df = pd.DataFrame({
    'EmployeeID': [f'EMP{str(i).zfill(4)}' for i in range(1, n+1)],
    'Age': age,
    'Gender': gender,
    'Education': education,
    'Department': department,
    'City': city,
    'Experience_Years': experience,
    'Salary': salary_col,
    'Performance_Score': performance,
    'Job_Satisfaction': satisfaction_col
})

df.to_csv('/home/claude/eda_project/data/employee_dataset.csv', index=False)
print(f"Dataset created: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head())
