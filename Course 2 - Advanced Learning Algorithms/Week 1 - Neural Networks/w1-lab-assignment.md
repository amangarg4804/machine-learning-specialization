# Understanding "Unrolling" a 20×20 Image into a 400-Dimensional Vector

## 1. The raw image
- Each digit (0 or 1) is stored as a **20×20 grid of pixels**.
- Think of it like a small black-and-white photo:  

20 rows
20 columns
total pixels = 20 × 20 = 400


- Each pixel has a **grayscale intensity**:
    - `0.0` = white
    - `1.0` = black
    - values in between = shades of gray

So one image looks like a **matrix**:

[ [p11, p12, p13, ..., p1,20],
[p21, p22, p23, ..., p2,20],
...
[p20,1, p20,2, ..., p20,20] ]


where each `pij` is the intensity of one pixel.

---

## 2. “Unrolling” into a vector
- Most ML algorithms don’t directly work with **matrices (20×20)**.
- Instead, they want a **single list of numbers (a vector)**.
- So we **flatten** the 20×20 matrix into a **400-dimensional vector**.

### Example (smaller 3×3 image for clarity):

Matrix:
[ [1, 2, 3],
[4, 5, 6],
[7, 8, 9] ]


Flatten (row by row) →  

[1, 2, 3, 4, 5, 6, 7, 8, 9]


- Now it’s a **1D vector of length 9**.
- For a 20×20 image, you’d get a **1D vector of length 400**.

---

## 3. The full dataset
- You have **1000 such images**.
- Each image → 400-dimensional vector.
- Stack them together to get:  

X.shape = (1000, 400)


This means:
- **1000 rows** → one row per image.
- **400 columns** → pixel intensities of that image.

---

## Why do this?
Because algorithms like **logistic regression** or **neural networks** expect inputs as vectors, not matrices. Flattening turns the image into the right input format.


# Why is W shaped (s_in, s_out) and b shaped (s_out,)?

Suppose:
- Current layer has `s_in` input units
- Next layer has `s_out` output units

We want to compute the activations of the next layer:

z = X · W + b
a = activation(z)


---

## 1. Input to the layer
- Let a **single example** (one row of `X`) have shape `(s_in,)`.  
  Example: for the input layer, `s_in = 400`.

---

## 2. Weight Matrix W
- Every output unit needs a **weighted sum of all input units**.
- So for each of the `s_out` outputs, we need a set of `s_in` weights.
- This means:
    - Number of rows = `s_in`
    - Number of columns = `s_out`
- Hence, `W` has shape `(s_in, s_out)`.

👉 Each column of `W` contains the weights for one output neuron.

---

## 3. Bias Vector b
- Each output neuron also has **one bias term**.
- Since there are `s_out` neurons, we need `s_out` biases.
- So `b` has shape `(s_out,)`.

---

## 4. Dimensions Check
- Input `x`: `(s_in,)`
- Weights `W`: `(s_in, s_out)`
- Bias `b`: `(s_out,)`

Matrix multiplication:

x (1, s_in) · W (s_in, s_out) = (1, s_out)


Then adding `b (s_out,)` → result is `(1, s_out)` → the correct number of outputs.

---

## Example
Say `s_in = 3`, `s_out = 2`.

- Input `x = [x1, x2, x3]` (shape `(3,)`).
- Weight matrix `W` = shape `(3,2)`:

W = [[w11, w12],
[w21, w22],
[w31, w32]]


- Bias `b = [b1, b2]` (shape `(2,)`).

Computation:

z1 = x1w11 + x2w21 + x3w31 + b1
z2 = x1w12 + x2w22 + x3w32 + b2


So we get 2 outputs `(z1, z2)`.

---

👉 In short:

W is (s_in, s_out) so every input connects to every output.

b is (s_out,) because each output has its own bias.

# Why is `W` shaped `(s_in, s_out)` and `b` shaped `(s_out,)`?

---

## Step 1: One neuron
A single neuron computes:

\[
z = w_1x_1 + w_2x_2 + w_3x_3 + b
\]

- Suppose the input layer has **3 inputs** (`x1, x2, x3`) → `s_in = 3`
- Suppose the next layer has **2 output neurons** → `s_out = 2`

That means we want two equations:

\[
z_1 = w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + b_1
\]  
\[
z_2 = w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + b_2
\]

---

## Step 2: Arrange weights into a matrix
- For **neuron 1** we need 3 weights: \((w_{11}, w_{21}, w_{31})\)
- For **neuron 2** we need 3 weights: \((w_{12}, w_{22}, w_{32})\)

Put them side by side:

W = [[w11, w12],
[w21, w22],
[w31, w32]]


- 3 rows (because 3 inputs)
- 2 columns (because 2 outputs)

👉 So `W.shape = (3, 2)` = `(s_in, s_out)`

---

## Step 3: Arrange biases
Each output neuron has its own bias:

b = [b1, b2]


- 2 numbers (because 2 outputs)  
  👉 So `b.shape = (2,)` = `(s_out,)`

---

## Step 4: Dimensions check
- Input `x = [x1, x2, x3]` → shape `(3,)`
- Multiply by `W`:

(1, 3) dot (3, 2) = (1, 2)


- Add `b (2,)` → final output = `(1, 2)`

✅ Correct: 3 inputs produce 2 outputs.

---

## Step 5: NumPy Demo
```python
import numpy as np

# 3 inputs
x = np.array([1.0, 2.0, 3.0])   # shape (3,)

# Weight matrix for 2 outputs
W = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6]])      # shape (3,2)

# Bias for 2 outputs
b = np.array([0.1, -0.1])       # shape (2,)

z = x @ W + b
print("z =", z)        
print("W.shape =", W.shape)
print("b.shape =", b.shape)

Output:
z = [2.3 3.1]
W.shape = (3, 2)
b.shape = (2,)
