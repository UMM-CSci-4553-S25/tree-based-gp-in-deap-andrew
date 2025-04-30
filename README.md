[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DClpi-Xf)

# Tree-Based Genetic Programming: Small or Large Classification

This project implements a tree-based Genetic Programming (GP) solution to the "Small or Large" classification benchmark problem using the DEAP framework in Python.

## 🔍 Problem Description

The goal is to evolve arithmetic programs that can determine whether a given number is considered "small" or "large" based on threshold boundaries, **without the program being told the threshold directly**. Some numbers fall in a "middle zone" and are not used for fitness evaluation, similar to how the original *Small or Large* problem is defined in the PSB1 benchmark suite by Lee Spector.

### ✅ Training Data Rules

- For **Dataset A**:
  - `n < 10` → `"small"`
  - `n ≥ 20` → `"large"`
  - `10 ≤ n < 20` → `"Middle Zone"` (ignored during fitness evaluation)

- For **Dataset B**:
  - `n < 20` → `"small"`
  - `n ≥ 30` → `"large"`
  - `20 ≤ n < 30` → `"Middle Zone"` (ignored during fitness evaluation)

To switch between two Training Data:
- You can toggle between Dataset A and B using the `USE_DATASET_A` flag in the code.

```bash
USE_DATASET_A = True #Set this to False if want to use Dataset B
train_data = train_data_a if USE_DATASET_A else train_data_b
```


---

## 🧠 Capture after each run

**Each experimental run outputs:**

- The best individual evolved (program tree structure).

- The fitness score of the best individual (number of misclassifications)

- The program’s predicted label

- The expected label

- A fitness graph that tracks the best fitness value across generations using matplotlib.

---

## Some example of output 

**Data A** 
```bash
gen	nevals	avg 	min
0  	100   	3.02	2  
1  	67    	2.93	2  
2  	65    	2.81	1  
3  	54    	2.61	0  
4  	62    	2.4 	0  
5  	76    	2.19	0  
6  	62    	1.91	0  
7  	53    	1.66	0  
8  	70    	1.44	0  
9  	59    	1.14	0  
10 	60    	1.11	0  
11 	63    	0.92	0  
12 	64    	1.1 	0  
13 	57    	0.77	0  
14 	62    	0.73	0  
15 	55    	0.72	0  
16 	56    	0.8 	0  
17 	54    	0.76	0  
18 	61    	0.77	0  
19 	47    	0.59	0  
20 	55    	0.72	0  
21 	55    	0.7 	0  
22 	58    	0.73	0  
23 	60    	0.86	0  
24 	71    	0.73	0  
25 	65    	0.8 	0  
26 	57    	0.59	0  
27 	60    	0.51	0  
28 	60    	0.78	0  
29 	55    	0.44	0  
30 	51    	0.5 	0  
31 	58    	0.63	0  
32 	49    	0.61	0  
33 	63    	0.65	0  
34 	61    	0.85	0  
35 	59    	0.67	0  
36 	52    	0.88	0  
37 	66    	0.97	0  
38 	72    	0.88	0  
39 	58    	0.75	0  
40 	60    	0.7 	0  
41 	70    	0.83	0  
42 	67    	0.69	0  
43 	65    	0.55	0  
44 	56    	0.63	0  
45 	68    	0.85	0  
46 	60    	0.56	0  
47 	59    	0.62	0  
48 	58    	0.51	0  
49 	52    	0.44	0  
50 	59    	0.54	0  
51 	66    	0.77	0  
52 	54    	0.59	0  
53 	56    	0.46	0  
54 	45    	0.51	0  
55 	62    	0.82	0  
56 	64    	0.63	0  
57 	41    	0.53	0  
58 	66    	0.53	0  
59 	62    	0.55	0  
60 	59    	0.56	0  

Best individual: sub(mul(x, 1.0), mul(2.0, add(3.0, 3.0)))
Fitness (total errors): 0.0

  Input → Predicted → Expected
    0 →        small → small
    5 →        small → small
    9 →        small → small
   10 →  Middle Zone → Middle Zone
   15 →  Middle Zone → Middle Zone
   19 →  Middle Zone → Middle Zone
   20 →        large → large
   25 →        large → large
   30 →        large → large
```
![TrainingDataA.png](..%2F..%2FUsers%2Flammi%2FDownloads%2FTrainingDataA.png)

**Data B**

```bash
gen	nevals	avg	min
0  	100   	3  	3  
1  	60    	3  	3  
2  	57    	3  	3  
3  	47    	3  	3  
4  	63    	3  	3  
5  	68    	3  	3  
6  	63    	3  	3  
7  	56    	3  	3  
8  	73    	3  	3  
9  	67    	3  	3  
10 	71    	3  	3  
11 	60    	3  	3  
12 	62    	3  	3  
13 	55    	3  	3  
14 	68    	3  	3  
15 	63    	3  	3  
16 	72    	3  	3  
17 	60    	3  	3  
18 	62    	3  	3  
19 	45    	3  	3  
20 	61    	3  	3  
21 	62    	3  	3  
22 	60    	3  	3  
23 	60    	3  	3  
24 	62    	3  	3  
25 	62    	3  	3  
26 	61    	3  	3  
27 	64    	3.03	3  
28 	54    	3   	3  
29 	66    	3   	3  
30 	57    	3   	3  
31 	66    	3   	3  
32 	54    	3   	3  
33 	51    	3   	3  
34 	64    	3   	3  
35 	66    	3   	3  
36 	69    	3   	3  
37 	53    	3   	3  
38 	60    	2.99	2  
39 	56    	2.98	2  
40 	66    	2.96	2  
41 	47    	2.92	2  
42 	64    	2.9 	2  
43 	63    	2.92	2  
44 	78    	2.87	2  
45 	58    	2.72	2  
46 	65    	2.66	1  
47 	61    	2.58	2  
48 	58    	2.5 	2  
49 	60    	2.38	1  
50 	73    	2.4 	1  
51 	46    	2.21	1  
52 	62    	2.24	1  
53 	56    	1.9 	0  
54 	52    	1.78	0  
55 	55    	1.7 	0  
56 	52    	1.51	0  
57 	58    	1.55	0  
58 	48    	1.27	0  
59 	59    	1.25	0  
60 	57    	1.13	0  

Best individual: protected_div(-1.0, protected_div(1.0, mul(3.0, sub(sub(add(3.0, protected_div(3.0, 0.5)), -1.0), mul(0.5, x)))))
Fitness (total errors): 0.0

  Input → Predicted → Expected
   10 →        small → small
   15 →        small → small
   19 →        small → small
   20 →  Middle Zone → Middle Zone
   25 →  Middle Zone → Middle Zone
   29 →  Middle Zone → Middle Zone
   30 →        large → large
   35 →        large → large
   40 →        large → large
```
![TrainingDataB.png](..%2F..%2FUsers%2Flammi%2FDownloads%2FTrainingDataB.png)

---
## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install deap numpy matplotlib
    
   python Small_Or_Large.py


📁 Notes
- The evolved solution is printed after evolution.

- Middle zone values are excluded from fitness evaluation but shown for completeness.

- The threshold itself is not explicitly provided to the evolved program — it must be inferred from the training examples.

📚 References

https://dl.acm.org/doi/abs/10.1145/2739480.2754769?casa_token=Eu6vVIMLlUcAAAAA%3Ah6CuMIIHZFZVNDb4XPUBjoJmCk-5oJ2vgPo5YBJF7jRZI2EeV-YSQ02MsoX0dye1vtCxBzllkYa_gg

https://thelmuth.github.io/GECCO_2015_Benchmarks_Materials/

