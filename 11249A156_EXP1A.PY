import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.arange(1, 10, 2)
c = np.linspace(0, 1, 5)
d = np.zeros((2, 3))
e = np.ones((2, 2))
f = np.eye(3)
g = np.random.randint(1, 100, size=(3, 3))
print("Array a:", a)
print("Arange b:", b)
print("Linspace c:", c)
print("Zeros d:\n", d)
print("Ones e:\n", e)
print("Identity f:\n", f)
print("Random g:\n", g)

#arrays properties.
print("Shape:", g.shape)
print("Dimensions:", g.ndim)
print("Size:", g.size)
print("Data type:", g.dtype)

#3.reshapes&transposes
print("Reshaped:", g.reshape(1, 9))
print("Transpose:\n", g.T)
print("Flattened:", g.flatten())

print("\nAdd:", np.add(a, 2))
print("Multiply:", np.multiply(a, 3))
print("Square Root:", np.sqrt(a))
print("Power:", np.power(a, 2))
print("Absolute:", np.abs([-1, -2, 3]))

# 5️⃣ Statistical Functions
print("\nMean:", np.mean(a))
print("Median:", np.median(a))
print("Standard Deviation:", np.std(a))
print("Variance:", np.var(a))
print("Minimum:", np.min(a))
print("Maximum:", np.max(a))
print("Sum:", np.sum(a))
print("Percentile (50):", np.percentile(a, 50))

#6.
print("\nColumn-wise Sum:\n", np.sum(g, axis=0))
print("Row-wise Mean:\n", np.mean(g, axis=1))

#7️⃣ Indexing & Slicing
print("\nFirst element:", a[0])
print("Slice:", a[1:4])
print("2D Slice:\n", g[0:2, 1:3])

# 8️⃣ Boolean & Conditional
print("\nWhere > 50:\n", np.where(g > 50))
print("Any > 90:", np.any(g > 90))
print("All > 0:", np.all(g > 0))

# 9️⃣ Sorting & Searching
print("\nSorted a:", np.sort(a))
print("Index of max:", np.argmax(a))
print("Index of min:", np.argmin(a))
print("Unique elements:", np.unique([1, 2, 2, 3, 4, 4]))

# 🔟 Linear Algebra
X = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
print("\nDot Product: \n", np.dot(X, y))
print("Matrix Multiplication:\n", np.matmul(X, y))
print("Determinant:", np.linalg.det(X))
print("Inverse: \n", np.linalg.inv(X))

# 1️⃣1️⃣ Random Functions
np.random.seed(1)
rand_sample = np.random.choice(a, size=3)
print("\nRandom Choice:", rand_sample)

# 1️⃣2️⃣ Stack & Concatenate
h1=np.array([1,2,3])
h2=np.array([4,5,6])
print("\nConcatenate : ", np.concatenate((h1, h2)))
print("Vertical stack : \n", np.vstack((h1, h2)))
print("Horizontal stack : \n", np.hstack((h1, h2)))

# 1️⃣3️⃣ Save & Load
np.save("sample_array.npy",a)
loaded_array = np.load("sample_array.npy")
print("\nLoaded Array:", loaded_array)
