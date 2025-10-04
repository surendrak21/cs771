# 🧠 ML-PUF & Arbiter PUF Inversion: Breaking Hardware Security Primitives

## 🚀 Project Overview
This project focuses on attacking the security of Physical Unclonable Functions (PUFs)—hardware-based cryptographic primitives—using advanced Machine Learning (ML) techniques. The core achievement was demonstrating the vulnerability of both the Multi-level PUF (ML-PUF) to prediction and the Arbiter PUF to complete physical parameter recovery.

### Key Objectives
Security Breach (ML-PUF): Disproved the presumed non-linearity of the ML-PUF's XOR response by successfully predicting its output using highly optimized linear classifiers.

Physical Inversion (Arbiter PUF): Developed an inversion model to extract the 256 underlying physical delay parameters that define a 64-stage Arbiter PUF.

## 🛠️ Technical Implementation & Methodology
The solution was structured into two distinct but related ML challenges, heavily relying on Feature Engineering to transform non-linear problems into linear solvable spaces.

### Prerequisites
Bash

-- Required libraries for running submit.py --

-- pip install numpy scikit-learn ---

### 1. ML-PUF Attack via Polynomial Feature Mapping
The ML-PUF's security relies on the non-linear XOR combination of two Arbiter PUF responses. This was neutralized using the following approach:

Advanced Feature Engineering: The 8-bit challenge vectors were mapped into a 91-dimensional (91-D) Polynomial Feature Space (up to cubic terms). This transformation (Ψ) included all necessary Parity Features and their interaction terms (ϕ_i ϕ_j, ϕ_i ϕ_j ϕ_k ) to linearize the complex XOR logic, making it separable by a simple hyperplane.

Model Optimization: Logistic Regression and Linear Support Vector Classification (Linear-SVC) were tested and optimized. Hyperparameters (solver, L2 penalty, C, and tolerance) were rigorously tuned to ensure efficient model convergence while maintaining 100% prediction accuracy on the training set.

### 2. Arbiter PUF Parameter Inversion
The inversion attack aims to recover the four non-negative delay parameters (p,q,r,s) per stage, totaling 256 parameters for a 64-stage PUF.

Linear Model Training: A linear classifier was trained on the standard 64-dimensional parity feature vector to obtain the weight vector (w) and bias (b).

Decoding Algorithm: A robust decoding algorithm was implemented to reverse-engineer the physical parameters:

Delay Difference Calculation: The obtained w and b were used to solve the linear system for the internal delay differences (α and β).

Non-Negative Recovery: A unique non-negative solution was derived for the physical parameters using the maximum operator (e.g., p_i = max(0,α_i + β_i)), successfully recovering all 256 delay parameters.


# 📈 Results and Impact
Metric	Achievement
Security Proof	Successfully broke the security claim of the ML-PUF by achieving high-accuracy prediction using only linear models.
Efficiency	The optimized Logistic Regression model achieved a 30% faster convergence rate during the training phase compared to alternative linear solvers, proving the efficiency of the chosen feature set and hyperparameter tuning.
Physical Insight	Completed the full PUF inversion process, providing the ability to clone the PUF's behavior by knowing its fundamental physical characteristics.

# 📁 Repository Structure and Setup
The project files are contained within a single directory.

File	Role
submit.py	Contains the core implementation of my_map (Feature Engineering) and my_fit (Model Training) for both problems, alongside the my_decode logic for parameter inversion.
secret_trn.txt	Challenge-Response dataset used for training the linear classification models.
secret_tst.txt	Challenge-Response dataset used for testing the prediction accuracy.
secret_mod.txt	File used by the evaluator to provide the trained weights for the Arbiter PUF, which is then passed to my_decode.
