# Statistical-Computing-and-Reporting
1. Describe the following (6 marks)
Cross-validation
A resampling method to evaluate model generalization by splitting data into training and validation folds (e.g., k-fold CV). It gives an estimate of model performance on unseen data and helps pick hyperparameters.
Bootstrap
A resampling technique that draws repeated samples (with replacement) from the observed data to estimate the sampling distribution of a statistic (e.g., mean, standard error, confidence intervals).
Monte Carlo simulation
Using repeated random sampling to approximate numerical results — for example probability integrals, expected values, or performance metrics — by simulating many random scenarios and averaging outcomes.
________________________________________
2. Five types of data representation in a computer (5 marks)
1.	Bits / Binary — lowest-level representation (0/1).
2.	Integers — fixed- or arbitrary-precision integer types (e.g., 32-bit signed).
3.	Floating-point numbers — real numbers stored in IEEE 754 format (single/double precision).
4.	Character / String encodings — ASCII or UTF-8 encoding of text.
5.	Structured collections — arrays, lists, matrices, tables (e.g., NumPy arrays, pandas DataFrame) that group values and preserve order/shape.
(Each entry: name + short description counts.)
________________________________________
3. Data analysis as process chain — briefly describe any three processes (6 marks)
1.	Data collection / acquisition — gather raw data from sources (sensors, DBs, APIs, surveys). Goal: get relevant, representative data.
2.	Data cleaning / preprocessing — handle missing values, remove duplicates, correct types, normalize/scale. This step ensures quality and correctness for modeling.
3.	Modeling / analysis — apply statistical or machine learning models to find patterns, make predictions, or test hypotheses. Validate models and tune hyperparameters.
(You could also mention visualization, evaluation, reporting.)
________________________________________
4. Define the following (2+1+1+1 = 5 marks)
Algorithm (2 marks)
A finite sequence of well-defined instructions for solving a class of problems; typically has inputs, a set of steps, and outputs (e.g., sorting algorithm).
Debugging (1 mark)
The process of locating, diagnosing, and fixing errors (bugs) in software.
Program (1 mark)
A set of instructions written in a programming language that performs tasks when executed by a computer.
Data Frame (1 mark)
A 2-dimensional tabular data structure (rows & columns), where columns may have different types (e.g., pandas.DataFrame).
________________________________________
5. Order the functions by growth rate and indicate equal-growth (10 marks)
Given functions (interpreting ambiguous parts reasonably), ordering from smallest to largest as n → ∞:
1.	2n\frac{2}{n}n2 — decays to 0 (smallest for large n).
2.	Constants: 2, 372,\ 372, 37 — constant growth (same rate as each other).
3.	n\sqrt{n}n — sublinear.
4.	nnn — linear.
5.	nlog⁡log⁡nn \log\log nnloglogn — slightly larger than n for large n.
6.	nlog⁡nn \log nnlogn and nlog⁡(n2)n\log(n^2)nlog(n2) — note nlog⁡(n2)=2nlog⁡nn\log(n^2)=2n\log nnlog(n2)=2nlogn so same big-Theta as nlog⁡nn\log nnlogn (constant factor).
7.	nlog⁡2nn \log^2 nnlog2n — larger than nlog⁡nn\log nnlogn.
8.	n1.5=n3/2n^{1.5} = n^{3/2}n1.5=n3/2 — polynomial, between nlog⁡2nn\log^2 nnlog2n and n2n^2n2 for large n.
9.	n2n^2n2
10.	n2log⁡nn^2 \log nn2logn
11.	n3n^3n3
12.	Exponential: e.g. 2n/22^{n/2}2n/2 or 2n/22^n/22n/2 (interpreted as exponential forms) — grows fastest.
Same-growth notes:
•	nlog⁡(n2)n\log(n^2)nlog(n2) and nlog⁡nn\log nnlogn are the same big-Theta (constant factor 2 difference).
•	Constants (2 and 37) have the same growth class (Θ(1)).
(If any term was intended differently in the exam, adjust accordingly — but this ordering is standard.)
8. The data contained in pandas objects can be assembled in different ways — describe three ways and show Python commands (6 marks)
1.	Construct from dictionaries / columns
import pandas as pd
df = pd.DataFrame({
    "id": [1,2,3],
    "name": ["a","b","c"],
    "value": [10.0, 20.0, 30.0]
})
2.	Concatenate multiple DataFrames (row-wise or column-wise)
df1 = pd.DataFrame({"id":[1,2], "a":[10,20]})
df2 = pd.DataFrame({"id":[3,4], "a":[30,40]})
df_rows = pd.concat([df1, df2], axis=0, ignore_index=True)  # rows
# column-wise (axis=1) can assemble different series into one DF
3.	Merge / join on keys (relational assembly)
left = pd.DataFrame({"id":[1,2], "x":[10,20]})
right = pd.DataFrame({"id":[1,2], "y":[100,200]})
merged = left.merge(right, on="id", how="inner")
(Other ways: from NumPy arrays via pd.DataFrame(np_array), from CSV via pd.read_csv().)
9. A function tmpFn that takes a single argument xVec and returns vector f(x) over -3 < x < 3 (6 marks)
The piecewise function from the image (interpreted) is:
f(x)={x2+2x+3x<02x−0.5x20≤x<2x2+4x−7x≥2f(x)= \begin{cases} x^2 + 2x + 3 & x < 0 \\ 2x - 0.5x^2 & 0 \le x < 2 \\ x^2 + 4x - 7 & x \ge 2 \end{cases}f(x)=⎩⎨⎧x2+2x+32x−0.5x2x2+4x−7x<00≤x<2x≥2 
Here is a vectorized Python implementation:
import numpy as np

def tmpFn(xVec):
    """
    xVec : array-like (1D)
    returns : numpy array of same shape with f(x) piecewise
    """
    x = np.asarray(xVec)
    y = np.empty_like(x, dtype=float)

    # conditions
    mask1 = x < 0
    mask2 = (x >= 0) & (x < 2)
    mask3 = x >= 2

    y[mask1] = x[mask1]**2 + 2*x[mask1] + 3
    y[mask2] = 2*x[mask2] - 0.5 * x[mask2]**2
    y[mask3] = x[mask3]**2 + 4*x[mask3] - 7
    return y

# Example:
xs = np.linspace(-3, 3, 13)
print(xs)
print(tmpFn(xs))
This returns a NumPy array of f(x) evaluated at each x in xVec.

10. Briefly describe (8 marks)
I. Divide and Conquer
An algorithmic paradigm that breaks a problem into subproblems, solves subproblems recursively, and combines solutions (e.g., merge sort, quicksort).
II. Problem abstraction
Representing a real problem in simplified, essential terms (data structures & operations) so it can be solved algorithmically while hiding irrelevant details.
III. Hierarchical indexing
In data structures (e.g., pandas MultiIndex) — multi-level index providing hierarchical organization of rows/columns enabling grouped operations and easy slicing.
IV. Debugging
Systematic process of finding and fixing software defects; tools include print/debugger/tracing, unit tests, and logging.
6.	Describe & give an example of conditional statements (use Python) (4 marks)
I. if ... elif ... else — execute mutually exclusive branches.
x = 5
if x < 0:
    print("negative")
elif x == 0:
    print("zero")
else:
    print("positive")
II. while — loop while a condition is true.
n = 5
i = 0
while i < n:
    print(i)
    i += 1
II. while — loop while a condition is true.
n = 5
i = 0
while i < n:
    print(i)
    i += 1

