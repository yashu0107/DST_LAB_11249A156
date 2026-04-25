# 1.3 Matplotlib (Plotting Library)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8],
    "ExamScore": [35, 40, 50, 55, 65, 70, 78, 85]
}

# Create DataFrame
df = pd.DataFrame(data)

# -----------------------------
# All Plots Combined
# -----------------------------

# 1️⃣ Scatter Plot
plt.figure()
plt.scatter(df["StudyHours"], df["ExamScore"])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score (Scatter Plot)")
plt.show()

# 2️⃣ Line Plot
plt.figure()
plt.plot(df["StudyHours"], df["ExamScore"], marker='o')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Exam Score Progress (Line Plot)")
plt.show()

# 3️⃣ Histogram
plt.figure()
plt.hist(df["ExamScore"], bins=5)
plt.xlabel("Score Range")
plt.ylabel("Number of Students")
plt.title("Exam Score Distribution (Histogram)")
plt.show()

# 4️⃣ Bar Chart
plt.figure()
plt.bar(df["StudyHours"], df["ExamScore"])
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score (Bar Chart)")
plt.show()
