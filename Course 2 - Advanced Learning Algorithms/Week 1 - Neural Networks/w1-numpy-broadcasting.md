## 🧮 NumPy Broadcasting Example

```python
import numpy as np

a = np.array([1, 2, 3, 4]).reshape(-1, 1)  # shape: (4, 1)
b = np.array([1, 2, 3]).reshape(1, -1)     # shape: (1, 3)

print("a =\n", a)
print("b =\n", b)
print(f"(a + b).shape: {(a + b).shape}, \na + b = \n{a + b}")
```

### 🔍 Explanation

#### 1. Reshaping

- `a` is reshaped to a **column vector** of shape `(4, 1)`:
  ```
  [[1]
   [2]
   [3]
   [4]]
  ```

- `b` is reshaped to a **row vector** of shape `(1, 3)`:
  ```
  [[1 2 3]]
  ```

#### 2. Broadcasting

When adding `a` and `b`, NumPy uses **broadcasting**:

- `a`: shape `(4, 1)`
- `b`: shape `(1, 3)`

These shapes are broadcasted to `(4, 3)`:
- Each row of `a` is added to each column of `b`.

#### 3. Element-wise Addition Breakdown

```
a + b =
[[1+1  1+2  1+3] → [2 3 4]
 [2+1  2+2  2+3] → [3 4 5]
 [3+1  3+2  3+3] → [4 5 6]
 [4+1  4+2  4+3] → [5 6 7]]
```

#### 4. Final Output

```
(a + b).shape: (4, 3)
a + b =
[[2 3 4]
 [3 4 5]
 [4 5 6]
 [5 6 7]]
```
