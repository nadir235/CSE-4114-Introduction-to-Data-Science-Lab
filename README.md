# Data Science Projects: 20 End-to-End Notebooks

Twenty self-contained data science projects, each delivered as a Jupyter notebook (`.ipynb`) that walks through the complete workflow from problem statement to conclusion. Every notebook downloads its own dataset, cleans it, explores it, engineers features, trains and compares several models, and prints a written discussion and conclusion.

---

## Table of contents

1. [Quick start](#quick-start)
2. [What every notebook contains](#what-every-notebook-contains)
3. [Methodology shared by all projects](#methodology-shared-by-all-projects)
4. [Project index](#project-index)
5. [Project details](#project-details)
6. [Output files](#output-files)
7. [Adapting a notebook to your own dataset](#adapting-a-notebook-to-your-own-dataset)
8. [Troubleshooting](#troubleshooting)
9. [Notes on results](#notes-on-results)

---

## Quick start

**Requirements:** Python 3.9+ and the packages below.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

**Run locally**

```bash
jupyter notebook            # then open any 01_... to 20_... .ipynb file
```

Choose **Kernel > Restart & Run All** (or run the cells top to bottom). An internet connection is needed on the first run because each notebook downloads its dataset from a public URL.

**Run in Google Colab:** upload a notebook to Colab and run all cells. No code changes are needed. If a download fails, Colab opens a file-upload dialog.

**Run time:** most notebooks finish in 1 to 5 minutes on a laptop. The heavier ones (03 Wine, 09 Bank churn, 15 Medical cost, 19 Dry bean) can take 10 to 20 minutes on a 2-core machine because they tune several models with cross-validation.

---

## What every notebook contains

All 20 notebooks share the same structure, so once you understand one you understand all of them.

| Part | Cells | Purpose |
|------|-------|---------|
| Title | Markdown | Project name, dataset, ML task |
| 0. Imports and setup | Code | Libraries, random seed (`RS = 42`), output folder, plot style |
| Configuration and hooks | Code | `CONFIG` dictionary (title, data URLs, problem, objectives, tasks, discussion, conclusion) and dataset-specific functions such as `create_features` |
| Generic helpers | Code | `section`, `savefig`, `is_cat`, `all_targets` |
| 1 to 9 | One code cell per function | The reusable workflow functions (see below) |
| Run the full workflow | Short code cells | Calls each step in order and shows the output |

The "Run the full workflow" section at the bottom is what you execute:

```python
problem_statement()                 # 1. problem statement and objectives
df_raw   = load_data()              # 2-3. dataset description and collection
df_clean = clean_data(df_raw)       # 4. cleaning and preprocessing
df_eda   = run_eda(df_clean)        # 5. exploratory data analysis
df_fe    = engineer_features(df_eda)  # 6. feature engineering
results  = model_development(df_fe)   # 7-8. selection, modelling, evaluation
conclusion(results)                 # 9. results, discussion, conclusion
```

---

## Methodology shared by all projects

### 1. Problem statement and objectives
Prints the project title, the business or scientific question, and a numbered list of objectives taken from `CONFIG`.

### 2-3. Dataset description and data collection
Downloads the data from the URLs in `CONFIG["urls"]` (tries each in order), then reports shape, column types, numeric and categorical summaries, and a data dictionary. If every download fails, it falls back to a custom loader, then to a manual upload (Colab) or the `DATA_PATH` environment variable (local).

### 4. Data cleaning and preprocessing
- Drops identifier columns and constant columns.
- Standardises text (trims whitespace, converts empty strings to missing).
- Removes duplicate rows.
- Reports missing values and fills them: median for numbers, mode (or a configured value) for categories; columns with more than 60% missing are dropped after feature engineering.
- Removes rows with a missing target.
- Handles outliers with the 1.5 x IQR rule, capping values at the fences (winsorising), and plots before and after boxplots.

### 5. Exploratory data analysis
Target distribution, numeric distributions and skewness, categorical counts, relationships between features and the target, correlation heatmap (including Cramer's V for categorical pairs), and dataset-specific extra charts where a project defines them. All figures are shown inline and saved as PNG files.

### 6. Feature engineering
- Dataset-specific feature creation (for example title and family size for Titanic, BMI for Obesity).
- `log1p` transformation of strongly right-skewed non-negative features (skew greater than 1).
- One-hot encoding of categoricals and standardisation of numerics, both done **inside the model pipeline** so test data never leaks into the preprocessing fit.

### 7. Feature selection
- Optional correlation filter that drops one of each highly correlated pair.
- Embedded selection with `SelectFromModel` using a random forest (threshold = 0.25 x mean importance), applied inside each pipeline.
- Feature-importance chart (random forest importance, plus mutual information where used).

### 8. Model development and evaluation
- 80/20 train/test split (stratified for classification), `random_state = 42`.
- 5-fold cross-validation on the training set (3-fold for very large sets).
- Automatic `class_weight="balanced"` when the minority class is under 35%.

| Task | Models compared | Main metric | Other metrics |
|------|-----------------|-------------|---------------|
| Classification | Majority-class baseline, Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, SVM (RBF, if 6,000 rows or fewer) | F1 | Accuracy, precision, recall, ROC-AUC |
| Regression | Mean-predictor baseline, Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, Gradient Boosting | R2 | MAE, RMSE |
| Clustering | K-Means, Agglomerative (Ward), DBSCAN, Gaussian Mixture | Silhouette | Davies-Bouldin, Calinski-Harabasz, PCA plots |

The best candidate models are tuned with `RandomizedSearchCV` (small grids, 3-fold). The best model is chosen by cross-validation score (not by test score) and then evaluated on the held-out test set with a confusion matrix, ROC curve, residual plots (regression) and permutation importance.

### 9. Results, discussion and conclusion
Prints the best model and its test metrics next to the baseline, the most influential features, a discussion of what the results mean, limitations and future work, and a one-paragraph conclusion. A `results_summary.csv` is saved.

---

## Project index

| # | Notebook | Dataset | Task | Rows x cols (raw) |
|---|----------|---------|------|-------------------|
| 01 | [`01_titanic_survival_classification.ipynb`](#project-1-titanic-survival-prediction) | Titanic - Machine Learning from Disaster (891 passengers) | Classification | 891 x 12 |
| 02 | [`02_iris_species_classification.ipynb`](#project-2-iris-species-classification) | Iris Species (150 flowers, 3 species) | Classification | 150 x 5 |
| 03 | [`03_wine_quality_classification_regression.ipynb`](#project-3-wine-quality-prediction-regression-and-classification) | Wine Quality - red (1,599) and white (4,898) Vinho Verde wines | Regression + Classification | 6,497 x 14 |
| 04 | [`04_heart_disease_classification.ipynb`](#project-4-heart-disease-prediction) | Heart Disease UCI - Cleveland subset (303 patients) | Classification | 303 x 14 |
| 05 | [`05_pima_diabetes_classification.ipynb`](#project-5-diabetes-onset-prediction-pima-indians) | Pima Indians Diabetes Database (768 women, 8 diagnostic measurements) | Classification | 768 x 9 |
| 06 | [`06_breast_cancer_wisconsin_classification.ipynb`](#project-6-breast-cancer-diagnosis-wisconsin-diagnostic) | Breast Cancer Wisconsin (Diagnostic) - 569 fine-needle-aspirate samples, 30 features | Classification | 569 x 33 |
| 07 | [`07_student_performance_regression_classification.ipynb`](#project-7-student-performance-prediction) | Students Performance in Exams (1,000 students) | Regression + Classification | 1,000 x 10 |
| 08 | [`08_mall_customers_clustering.ipynb`](#project-8-mall-customer-segmentation) | Mall Customers (200 customers) | Clustering | 200 x 5 |
| 09 | [`09_bank_customer_churn_classification.ipynb`](#project-9-bank-customer-churn-prediction) | Bank Customer Churn / Churn_Modelling (10,000 customers) | Classification | 10,000 x 14 |
| 10 | [`10_telco_customer_churn_classification.ipynb`](#project-10-telco-customer-churn-prediction) | Telco Customer Churn (7,043 customers) | Classification | 7,043 x 21 |
| 11 | [`11_car_price_prediction_regression.ipynb`](#project-11-car-price-prediction) | Car Price Prediction / CarPrice_Assignment (205 cars) | Regression | 205 x 26 |
| 12 | [`12_laptop_price_regression.ipynb`](#project-12-laptop-price-prediction) | Laptop Price Dataset (1,303 laptops) | Regression | 1,303 x 12 |
| 13 | [`13_used_cars_regression.ipynb`](#project-13-used-car-selling-price-prediction) | Used Cars Dataset / CarDekho (301 listings) | Regression | 301 x 9 |
| 14 | [`14_insurance_charges_regression.ipynb`](#project-14-insurance-charges-prediction) | Insurance Charges Dataset (1,338 policyholders) | Regression | 1,338 x 7 |
| 15 | [`15_medical_cost_personal_regression.ipynb`](#project-15-medical-cost-prediction-and-high-cost-patient-detection) | Medical Cost Personal Datasets (1,338 beneficiaries) | Regression + Classification | 1,338 x 8 |
| 16 | [`16_spotify_tracks_classification_clustering.ipynb`](#project-16-spotify-tracks-genre-classification-and-audio-clustering) | Spotify Tracks / Spotify Songs (about 28,000 unique tracks, 6 genres) | Classification + Clustering | 28,356 x 23 |
| 17 | [`17_netflix_eda_recommendation.ipynb`](#project-17-netflix-catalogue-eda-and-content-based-recommender) | Netflix Movies and TV Shows (7,787 titles, to 2021) | EDA + Recommendation + Clustering | 7,787 x 14 |
| 18 | [`18_mushroom_classification.ipynb`](#project-18-mushroom-edibility-classification) | Mushroom (8,124 records, 22 categorical attributes) | Classification | 8,124 x 23 |
| 19 | [`19_dry_bean_classification.ipynb`](#project-19-dry-bean-variety-classification) | Dry Bean Dataset (13,611 beans, 7 varieties, 16 shape features) | Classification | 13,611 x 17 |
| 20 | [`20_obesity_levels_classification.ipynb`](#project-20-obesity-level-prediction) | Estimation of Obesity Levels Based on Eating Habits and Physical Condition (2,111 people) | Classification | 2,135 x 17 |

---

## Project details

### Project 1: Titanic survival prediction

**Notebook:** `01_titanic_survival_classification.ipynb`  
**Dataset:** Titanic - Machine Learning from Disaster (891 passengers)  
**Source:** Kaggle competition 'titanic'; CSV mirror: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv  
**Output folder:** `outputs/01_titanic/`

**About the data.** Passenger records from the 1912 sinking of the RMS Titanic with demographics, ticket and cabin details and whether the passenger survived.

**Problem statement.** Predict whether a passenger survived the Titanic disaster from information known at boarding (class, sex, age, family on board, fare, cabin, port).

**Objectives**

1. Clean the data (missing age, cabin and port values, duplicates, fare outliers).
2. Explore which passenger groups had the best survival chances.
3. Engineer informative features (title, family size, deck, fare per person).
4. Train and compare several classifiers and report accuracy, precision, recall, F1 and ROC-AUC.

**Key columns**

| Column | Meaning |
|--------|---------|
| `Survived` | target, 1 = survived |
| `Pclass` | ticket class 1-3 |
| `SibSp/Parch` | siblings-spouses / parents-children aboard |
| `Fare` | ticket price |
| `Cabin` | cabin number (mostly missing) |
| `Embarked` | port of embarkation |

**Modelling tasks**

- **Survival** (classification; target `Survived`)

**Results from the test run**

- Survival: best model = Logistic Regression  (test F1 = 0.748, accuracy = 0.816, ROC-AUC = 0.856; majority-class baseline accuracy = 0.615)
  - Top features: Title, Sex, Age, HasCabin

**Discussion**

- Sex, passenger class and the extracted title are expected to dominate: 'women and children first' and wealth are visible in the data.
- Family size has a non-linear effect: passengers travelling alone or in very large groups survived less often than small families.
- Logistic regression won on cross-validation, so on this small tabular set a simple, well-featured linear model matches or beats tree ensembles; trees are competitive but overfit more.

**Limitations and future work**

- Only 891 rows: results vary with the random split, so repeated cross-validation would give tighter estimates.
- Age is imputed with the median; imputing by title group could be more accurate.
- Ticket and surname groups (families travelling together) were not exploited.

**Conclusion.** The pipeline turns raw passenger records into a reliable survival classifier that clearly beats the majority-class baseline; the most important drivers are sex, title and class, matching the historical account of the evacuation.

---

### Project 2: Iris species classification

**Notebook:** `02_iris_species_classification.ipynb`  
**Dataset:** Iris Species (150 flowers, 3 species)  
**Source:** Fisher (1936) / UCI; CSV copy: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv  
**Output folder:** `outputs/02_iris/`

**About the data.** Sepal and petal measurements of 150 iris flowers from the species setosa, versicolor and virginica.

**Problem statement.** Identify the species of an iris flower from four measurements of its sepals and petals.

**Objectives**

1. Check and clean the measurements (duplicates, outliers).
2. Visualise how well the species separate in feature space.
3. Create geometric features (areas and ratios).
4. Compare classifiers with cross-validation and report per-class performance.

**Key columns**

| Column | Meaning |
|--------|---------|
| `sepal_length/width` | cm |
| `petal_length/width` | cm |
| `species` | target |

**Modelling tasks**

- **Species** (classification; target `species`)

**Results from the test run**

- Species: best model = Logistic Regression (tuned)  (test F1 = 0.933, accuracy = 0.933, ROC-AUC = 0.997; majority-class baseline accuracy = 0.333)
  - Top features: petal_area, petal_width, petal_length, sepal_ratio

**Discussion**

- Setosa is linearly separable; versicolor and virginica overlap slightly, which causes the few errors seen in the confusion matrix.
- Petal measurements carry almost all the information, so feature selection removes sepal-based features first.
- Simple models (logistic regression, KNN, SVM) perform as well as ensembles because the classes are almost separable.

**Limitations and future work**

- With 150 samples the test set has only 30 flowers, so one error changes accuracy by 3%; use repeated stratified CV for stable numbers.

**Conclusion.** Species can be identified with very high accuracy from petal size; the exercise shows that a small, clean dataset favours simple models.

---

### Project 3: Wine quality prediction (regression and classification)

**Notebook:** `03_wine_quality_classification_regression.ipynb`  
**Dataset:** Wine Quality - red (1,599) and white (4,898) Vinho Verde wines  
**Source:** Cortez et al. (2009), UCI Machine Learning Repository; CSV copies: https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv and https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-white.csv  
**Output folder:** `outputs/03_wine/`

**About the data.** Physicochemical laboratory measurements of Portuguese Vinho Verde wines with a sensory quality score (0-10) given by tasters.

**Problem statement.** Predict the sensory quality of a wine from objective laboratory measurements, both as a numeric score and as a quality class.

**Objectives**

1. Merge red and white wines and remove duplicate lab records.
2. Study how alcohol, acidity and sulphur compounds relate to quality.
3. Engineer chemistry-based ratios.
4. Solve the problem as regression (score) and as classification (Low <=5, Medium 6, High >=7) and compare models.

**Modelling tasks**

- **Quality score (regression)** (regression; target `quality`)
- **Quality class (classification)** (classification; target `quality_label`)

**Results from the test run**

- Quality score (regression): best model = Random Forest  (test R2 = 0.413, MAE = 0.509, RMSE = 0.664; mean-predictor RMSE = 0.868)
  - Top features: alcohol, volatile acidity, alcohol_x_sulphates, free_sulfur_ratio
- Quality class (classification): best model = Random Forest  (test F1 = 0.628, accuracy = 0.629, ROC-AUC = 0.782; majority-class baseline accuracy = 0.437)
  - Top features: alcohol, volatile acidity, pH, density

**Discussion**

- Alcohol is the strongest single predictor; volatile acidity (vinegar taste) and sulphates also matter.
- Quality is a noisy human rating concentrated on 5-6, so R2 stays moderate and the 'High' class is hard to recall: this is a data limit, not only a model limit.
- Random forests and boosting capture the non-linear chemistry better than linear models.

**Limitations and future work**

- Ordinal regression or cost-sensitive learning could respect the ordering of quality levels.
- Class imbalance for 'High' wines could be addressed with resampling (SMOTE).

**Conclusion.** Objective chemistry explains a useful share of wine quality; ensemble models give the best results but the subjectivity of tasting caps achievable accuracy.

---

### Project 4: Heart disease prediction

**Notebook:** `04_heart_disease_classification.ipynb`  
**Dataset:** Heart Disease UCI - Cleveland subset (303 patients)  
**Source:** UCI Machine Learning Repository (Detrano et al.); CSV copy: https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv  
**Output folder:** `outputs/04_heart/`

**About the data.** Clinical measurements (age, blood pressure, cholesterol, ECG results, exercise test) of patients, with an indicator of heart disease.

**Problem statement.** Predict from routine clinical measurements whether a patient has heart disease, to support early screening.

**Objectives**

1. Clean the clinical data and treat coded columns (cp, thal, slope ...) as categories.
2. Explore the risk factors visually.
3. Create clinically motivated features (heart-rate reserve, age group).
4. Compare classifiers with emphasis on recall (missing a sick patient is costly).

**Key columns**

| Column | Meaning |
|--------|---------|
| `cp` | chest pain type |
| `trestbps` | resting blood pressure |
| `chol` | cholesterol |
| `thalach` | max heart rate |
| `oldpeak` | ST depression |
| `ca` | vessels coloured by fluoroscopy |
| `thal` | thalassemia code |
| `target` | 1 = heart disease (as coded in this file) |

**Modelling tasks**

- **Heart disease** (classification; target `target`)

**Results from the test run**

- Heart disease: best model = Logistic Regression  (test F1 = 0.870, accuracy = 0.852, ROC-AUC = 0.905; majority-class baseline accuracy = 0.541)
  - Top features: ca, sex, thal, cp

**Discussion**

- Number of coloured vessels (ca), thalassemia result, sex and chest-pain type were the most influential features in the final model.
- For screening, recall matters more than precision; the decision threshold could be lowered to catch more cases at the cost of false alarms.
- Only 303 patients: tuning gains are small and test metrics are noisy.

**Limitations and future work**

- Validate on the other UCI sites (Hungary, Switzerland, VA) for external validity.
- Add calibration curves and threshold optimisation for a clinical operating point.

**Conclusion.** Standard clinical variables allow good discrimination of patients with heart disease; the model is a decision-support prototype, not a diagnostic tool.

---

### Project 5: Diabetes onset prediction (Pima Indians)

**Notebook:** `05_pima_diabetes_classification.ipynb`  
**Dataset:** Pima Indians Diabetes Database (768 women, 8 diagnostic measurements)  
**Source:** National Institute of Diabetes and Digestive and Kidney Diseases; CSV copy: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv  
**Output folder:** `outputs/05_pima/`

**About the data.** Diagnostic measurements of Pima Indian women aged 21+ and whether they developed diabetes within five years.

**Problem statement.** Predict whether a patient will develop diabetes from basic diagnostic measurements.

**Objectives**

1. Detect biologically impossible zeros (glucose, blood pressure, BMI ...) and treat them as missing.
2. Explore how glucose, BMI and age relate to diabetes.
3. Engineer clinical interaction features.
4. Compare models with attention to recall of the diabetic class.

**Key columns**

| Column | Meaning |
|--------|---------|
| `Outcome` | target, 1 = diabetes |
| `Insulin` | 2-hour serum insulin |
| `DiabetesPedigreeFunction` | family-history score |

**Modelling tasks**

- **Diabetes** (classification; target `Outcome`)

**Results from the test run**

- Diabetes: best model = Logistic Regression  (test F1 = 0.661, accuracy = 0.740, ROC-AUC = 0.812; majority-class baseline accuracy = 0.649)
  - Top features: AgeGroup, BMI, Glucose, Pregnancies

**Discussion**

- Age group, BMI, glucose and number of pregnancies were the most influential features; glucose and BMI are the clearest medical risk factors.
- Zeros in insulin and skin thickness are missing measurements; median imputation keeps all rows but adds uncertainty.
- The data are imbalanced (about 35% positive) so F1 and recall are more informative than accuracy.

**Limitations and future work**

- Model-based imputation (KNN / iterative) could handle the many missing insulin values better.
- The cohort is a single population, so the model may not generalise to others.

**Conclusion.** Simple screening measurements give a moderately accurate diabetes classifier; glucose and BMI drive the prediction.

---

### Project 6: Breast cancer diagnosis (Wisconsin Diagnostic)

**Notebook:** `06_breast_cancer_wisconsin_classification.ipynb`  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) - 569 fine-needle-aspirate samples, 30 features  
**Source:** UCI Machine Learning Repository (Wolberg et al.); CSV copy: https://raw.githubusercontent.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/master/data.csv  
**Output folder:** `outputs/06_breast_cancer/`

**About the data.** Features computed from digitised images of breast-mass cell nuclei (radius, texture, concavity ... as mean, standard error and worst value).

**Problem statement.** Classify a breast mass as malignant or benign from cell-nucleus measurements.

**Objectives**

1. Clean the data and remove the empty column.
2. Study the strong multicollinearity between size-related features.
3. Select a compact feature set (correlation filter + model-based selection).
4. Compare classifiers with focus on recall of malignant tumours.

**Modelling tasks**

- **Diagnosis** (classification; target `diagnosis`)

**Results from the test run**

- Diagnosis: best model = Gradient Boosting  (test F1 = 0.950, accuracy = 0.965, ROC-AUC = 0.997; majority-class baseline accuracy = 0.632)
  - Top features: concave points_mean, radius_mean, texture_worst, radius_worst_to_mean

**Discussion**

- Size and shape of the nucleus (worst radius, perimeter, concave points) separate malignant from benign samples almost linearly.
- Many features are >0.95 correlated; removing redundant ones barely changes accuracy and improves interpretability.
- In a medical setting a false negative (missed cancer) is far worse than a false positive, so recall of the malignant class is the key figure.

**Limitations and future work**

- Use the decision threshold chosen on validation data to guarantee a minimum recall.
- Check generalisation on data from other hospitals and imaging devices.

**Conclusion.** A handful of nucleus features is enough for a highly accurate diagnostic classifier; simple models (logistic regression, SVM) matched or beat the ensembles on the test set, so model complexity is not needed here.

---

### Project 7: Student performance prediction

**Notebook:** `07_student_performance_regression_classification.ipynb`  
**Dataset:** Students Performance in Exams (1,000 students)  
**Source:** Kaggle 'Students Performance in Exams'; CSV copy: https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv  
**Output folder:** `outputs/07_student/`

**About the data.** Exam scores (math, reading, writing) of 1,000 students with gender, parental education, lunch type and test-preparation course.

**Problem statement.** Predict a student's math score from background and other scores (regression) and flag students at risk of a low average score using background information only (classification).

**Objectives**

1. Explore how socio-economic factors relate to exam results.
2. Create average score and at-risk indicators.
3. Regression: predict the math score. Classification: predict 'At risk' (average < 60) from background features only.
4. Compare models and discuss fairness-relevant findings.

**Key columns**

| Column | Meaning |
|--------|---------|
| `lunch` | standard or free/reduced (socio-economic proxy) |
| `test preparation course` | completed or none |

**Modelling tasks**

- **Math score (regression)** (regression; target `math score`; excludes `average_score`)
- **At-risk student (classification)** (classification; target `result`; excludes `reading score`, `writing score`, `average_score`)

**Results from the test run**

- Math score (regression): best model = Ridge Regression (tuned)  (test R2 = 0.850, MAE = 4.678, RMSE = 6.036; mean-predictor RMSE = 15.732)
  - Top features: writing score, gender, reading score, race/ethnicity
- At-risk student (classification): best model = SVM (RBF)  (test F1 = 0.479, accuracy = 0.575, ROC-AUC = 0.698; majority-class baseline accuracy = 0.715)
  - Top features: is_prepared, free_lunch, gender, test preparation course

**Discussion**

- Reading and writing scores predict the math score very well (R2 is high) because academic skills are strongly correlated.
- Using background information only, predicting who is at risk is much harder: test preparation and lunch type matter, but ROC-AUC is only about 0.7 and accuracy falls below the majority-class baseline because class weighting trades accuracy for recall of at-risk students.
- Features such as gender and ethnicity group are sensitive; they should not be used for real decisions about students.

**Limitations and future work**

- Add attendance, study time and previous grades for a realistic early-warning system.
- Audit the classifier for bias across groups before any deployment.

**Conclusion.** Prior scores explain most of the math result, while background features alone give only modest early-warning ability; the study highlights the role of preparation and support.

---

### Project 8: Mall customer segmentation

**Notebook:** `08_mall_customers_clustering.ipynb`  
**Dataset:** Mall Customers (200 customers)  
**Source:** Kaggle 'Mall Customer Segmentation Data'; CSV copy: https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv  
**Output folder:** `outputs/08_mall/`

**About the data.** Gender, age, annual income (k$) and a 1-100 spending score assigned by the mall for 200 customers.

**Problem statement.** Group mall customers into meaningful segments so that marketing campaigns can be targeted.

**Objectives**

1. Clean the data and explore age, income and spending distributions.
2. Find the number of segments using the elbow method and silhouette analysis.
3. Compare K-Means, hierarchical, Gaussian-mixture and DBSCAN clustering.
4. Profile every segment and suggest a marketing action for it.

**Modelling tasks**

- **Customer segments** (clustering; features `Age`, `Annual Income (k$)`, `Spending Score (1-100)`)

**Results from the test run**

- Customer segments: best method = K-Means with k = 8 (silhouette = 0.430, Davies-Bouldin = 0.780)

**Discussion**

- Income and spending score form clearly separated groups (e.g. high income / high spending versus high income / low spending); age adds a younger, high-spending segment.
- K-Means obtained the best silhouette, ahead of the Gaussian mixture and Ward clustering; DBSCAN found no setting with at least two clusters and under 30% noise, so it is unsuitable for this evenly dense data. Silhouette favours a fairly large k (8) on three features; a smaller k such as 5 may be easier to act on commercially.
- Segment ideas: loyalty offers for high-income high-spenders, targeted promotions to convert high-income low-spenders, budget deals for low-income high-spenders.

**Limitations and future work**

- Only 200 customers and three variables; purchase history would enable RFM-style segmentation.
- Segments are unlabelled, so validation relies on internal metrics and business judgment.

**Conclusion.** The customers split into a small number of well-separated segments with distinct income-spending behaviour; the profiles can directly drive targeted marketing.

---

### Project 9: Bank customer churn prediction

**Notebook:** `09_bank_customer_churn_classification.ipynb`  
**Dataset:** Bank Customer Churn / Churn_Modelling (10,000 customers)  
**Source:** Kaggle 'Churn Modelling'; CSV copy: https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv  
**Output folder:** `outputs/09_bank_churn/`

**About the data.** Credit score, geography, gender, age, tenure, balance, products, card/activity flags and salary for 10,000 bank customers, with whether they left the bank.

**Problem statement.** Predict which bank customers are likely to leave so that retention actions can be targeted.

**Objectives**

1. Clean the data and drop identifiers (row number, customer id, surname).
2. Analyse churn by geography, age, products and activity.
3. Engineer financial ratios and age groups.
4. Handle class imbalance and compare models using F1, recall and ROC-AUC.

**Modelling tasks**

- **Churn** (classification; target `Exited`)

**Results from the test run**

- Churn: best model = Gradient Boosting  (test F1 = 0.597, accuracy = 0.864, ROC-AUC = 0.858; majority-class baseline accuracy = 0.796)
  - Top features: Age, NumOfProducts, Geography, IsActiveMember

**Discussion**

- Age, number of products, activity status and geography (Germany) drive churn; customers with 3-4 products churn at very high rates.
- About 20% of customers churn, so accuracy alone is misleading; class weighting and F1/ROC-AUC give a fairer picture.
- Gradient boosting and random forests reach the best balance of precision and recall.

**Limitations and future work**

- Use a cost-based threshold: the cost of a lost customer versus the cost of a retention offer.
- Transaction-level behaviour (balance trends, complaints) would improve prediction considerably.

**Conclusion.** Churn can be predicted with useful accuracy; the bank should prioritise older, inactive customers and those with many products or in Germany for retention campaigns.

---

### Project 10: Telco customer churn prediction

**Notebook:** `10_telco_customer_churn_classification.ipynb`  
**Dataset:** Telco Customer Churn (7,043 customers)  
**Source:** IBM sample data; CSV copy: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv  
**Output folder:** `outputs/10_telco_churn/`

**About the data.** Demographics, subscribed services, contract, billing and tenure information of telecom customers and whether they churned in the last month.

**Problem statement.** Predict which telecom customers will churn so the company can offer targeted retention incentives.

**Objectives**

1. Fix data-type problems (TotalCharges stored as text) and impute missing values.
2. Find churn drivers: contract type, tenure, internet service, payment method.
3. Create service-count and tenure-group features.
4. Compare models under class imbalance.

**Modelling tasks**

- **Churn** (classification; target `Churn`)

**Results from the test run**

- Churn: best model = Logistic Regression (tuned)  (test F1 = 0.614, accuracy = 0.740, ROC-AUC = 0.839; majority-class baseline accuracy = 0.735)
  - Top features: tenure, InternetService, MonthlyCharges, Contract

**Discussion**

- Month-to-month contracts, short tenure, fibre-optic internet and electronic-check payment are linked to high churn; long contracts protect against churn.
- Roughly one customer in four churns, so recall and F1 of the churn class are tracked in addition to accuracy.
- Logistic regression is competitive with ensembles, which makes it attractive because it is easy to explain to business users.

**Limitations and future work**

- Add usage and customer-service call data.
- Estimate customer lifetime value to prioritise which churners to target.

**Conclusion.** Contract type and tenure are the main churn levers; moving customers to longer contracts and supporting new fibre customers early should reduce churn.

---

### Project 11: Car price prediction

**Notebook:** `11_car_price_prediction_regression.ipynb`  
**Dataset:** Car Price Prediction / CarPrice_Assignment (205 cars)  
**Source:** Kaggle 'Car Price Prediction' (Geely Auto assignment); CSV copy: https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv  
**Output folder:** `outputs/11_car_price/`

**About the data.** Technical specifications (engine, dimensions, fuel system, mileage) and price of 205 car models in the US market.

**Problem statement.** Estimate a car's price from its specifications to understand which factors drive pricing.

**Objectives**

1. Clean the data and extract the brand from the car name (fixing misspellings).
2. Explore price against engine size, horsepower, weight and body type.
3. Create power-to-weight, fuel-economy and volume features; log-transform the price.
4. Compare regression models using MAE, RMSE and R2.

**Modelling tasks**

- **Car price** (regression; target `price`)

**Results from the test run**

- Car price: best model = Random Forest  (test R2 = 0.957, MAE = 1284.480, RMSE = 1840.188; mean-predictor RMSE = 9092.756)
  - Top features: curbweight, enginesize, highwaympg, mpg_avg

**Discussion**

- Curb weight, engine size and fuel economy were the most influential features; heavier, bigger-engined, less fuel-efficient cars cost more.
- Price is right-skewed, so modelling log(price) stabilises the variance and improves linear models.
- With only 205 cars the brand effect is noisy; tree ensembles and regularised linear models perform best.

**Limitations and future work**

- More observations per brand would stabilise brand effects.
- Prices are from one market and year; time effects are not modelled.

**Conclusion.** Car price is largely explained by engine power and size, and brand adds a premium; the final model predicts price with a good R2 on unseen cars.

---

### Project 12: Laptop price prediction

**Notebook:** `12_laptop_price_regression.ipynb`  
**Dataset:** Laptop Price Dataset (1,303 laptops)  
**Source:** Kaggle 'Laptop Price'; CSV copy: https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv  
**Output folder:** `outputs/12_laptop_price/`

**About the data.** Brand, type, screen, CPU, RAM, storage, GPU, operating system, weight and price (INR) of 1,303 laptops.

**Problem statement.** Predict the price of a laptop from its hardware specification.

**Objectives**

1. Convert text specs (RAM '8GB', weight '1.3kg') to numbers.
2. Parse screen resolution, CPU, storage and GPU strings into useful features (PPI, touchscreen, IPS, SSD/HDD size, CPU brand/speed).
3. Explore price drivers and log-transform the target.
4. Compare regression models.

**Modelling tasks**

- **Laptop price** (regression; target `Price`)

**Results from the test run**

- Laptop price: best model = Random Forest (tuned)  (test R2 = 0.785, MAE = 11082.771, RMSE = 19174.194; mean-predictor RMSE = 43464.487)
  - Top features: Ram, TypeName, Weight, Cpu_GHz

**Discussion**

- RAM, laptop type, CPU speed and weight were the most influential features; the EDA plots also show gaming and workstation types priced highest.
- Prices are right-skewed, so log(price) is used and predictions are transformed back to rupees for the metrics.
- Tree ensembles capture interactions between brand, CPU and RAM and give the best R2.

**Limitations and future work**

- Release year and GPU model benchmarks would improve accuracy.
- Model errors are larger for premium laptops; a quantile or heteroscedastic model could help.

**Conclusion.** Hardware specifications allow accurate price estimation; the engineered features (PPI, SSD size, CPU class) add clear value over the raw text columns.

---

### Project 13: Used car selling-price prediction

**Notebook:** `13_used_cars_regression.ipynb`  
**Dataset:** Used Cars Dataset / CarDekho (301 listings)  
**Source:** Kaggle 'Vehicle dataset from CarDekho'; CSV copy: https://raw.githubusercontent.com/krishnaik06/Car-Price-Prediction/master/car%20data.csv  
**Output folder:** `outputs/13_used_cars/`

**About the data.** Name, year, present showroom price, kilometres driven, fuel, seller type, transmission, number of previous owners and selling price (lakh INR) of used cars.

**Problem statement.** Estimate the fair selling price of a used car from its age, usage and original price.

**Objectives**

1. Clean the listings and handle duplicates and outliers.
2. Study depreciation: price versus age, kilometres, fuel and transmission.
3. Create car age, usage-per-year and popularity features.
4. Compare regression models.

**Modelling tasks**

- **Used car selling price** (regression; target `Selling_Price`)

**Results from the test run**

- Used car selling price: best model = Random Forest  (test R2 = 0.779, MAE = 1.036, RMSE = 2.388; mean-predictor RMSE = 5.307)
  - Top features: Price_Retention_Proxy, Present_Price, Car_Name_freq, Car_Age

**Discussion**

- The present (new) price, together with the derived retention proxy, is the strongest predictor, followed by the car's popularity and kilometres driven; fuel type, transmission and seller type also differ in the EDA plots.
- Selling price is skewed; the log transformation reduces the influence of a few expensive cars.
- Dealer listings are priced higher than individual-seller listings for the same car.

**Limitations and future work**

- The dataset is small and old; a larger recent dataset would capture current market conditions.
- Model, trim level and condition are missing and would explain extra variance.

**Conclusion.** Used-car prices follow a predictable depreciation pattern anchored on the new price and age; the final model gives accurate estimates for typical cars.

---

### Project 14: Insurance charges prediction

**Notebook:** `14_insurance_charges_regression.ipynb`  
**Dataset:** Insurance Charges Dataset (1,338 policyholders)  
**Source:** Kaggle 'Insurance' (Machine Learning with R, Lantz); CSV copy: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv  
**Output folder:** `outputs/14_insurance/`

**About the data.** Age, sex, BMI, children, smoker status, region and the medical charges billed by the health insurer.

**Problem statement.** Predict the yearly insurance charge of a person from personal attributes to support premium setting.

**Objectives**

1. Clean the data and treat BMI outliers.
2. Quantify the effect of smoking, age and BMI on charges.
3. Create BMI category, age group and a smoker x obesity interaction.
4. Compare regression models on the log-transformed target.

**Modelling tasks**

- **Insurance charges** (regression; target `charges`)

**Results from the test run**

- Insurance charges: best model = K-Nearest Neighbours  (test R2 = 0.894, MAE = 2139.307, RMSE = 4405.193; mean-predictor RMSE = 14588.308)
  - Top features: smoker_obese, smoker, age_squared, age

**Discussion**

- Smoking is by far the largest cost driver, followed by age; obesity multiplies the smoker effect (visible in the interaction feature).
- Charges have a long right tail; modelling log(charges) helps linear models, while distance- and tree-based models pick up the smoker x BMI interaction; K-Nearest Neighbours obtained the best cross-validation score here.
- Region and sex have little effect after controlling for smoking, age and BMI.

**Limitations and future work**

- Medical history is missing; it would explain residual variance.
- Premium fairness and regulation constraints are not considered.

**Conclusion.** Smoking status, age and BMI explain most variation in charges, and the best model predicts costs with high R2, which can inform risk-based pricing.

---

### Project 15: Medical cost prediction and high-cost patient detection

**Notebook:** `15_medical_cost_personal_regression.ipynb`  
**Dataset:** Medical Cost Personal Datasets (1,338 beneficiaries)  
**Source:** Kaggle 'Medical Cost Personal Datasets' (Lantz); CSV copy: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv  
**Output folder:** `outputs/15_medical_cost/`

**About the data.** Age, sex, BMI, number of children, smoker status, region and individual medical costs billed by health insurance. (This is the same underlying file as project 14; this project takes a different angle: raw-scale regression with polynomial/interaction features, plus a high-cost classification task.)

**Problem statement.** Predict individual medical costs in currency units, and flag beneficiaries in the top 25% of costs so that care management can focus on them.

**Objectives**

1. Clean the data and analyse cost distribution by smoker status, BMI and age.
2. Regression on the raw cost scale with polynomial and interaction features.
3. Classification of 'high-cost' beneficiaries (charges above the 75th percentile).
4. Compare models for both tasks and discuss business use.

**Modelling tasks**

- **Medical charges (regression, raw scale)** (regression; target `charges`; excludes `high_cost`)
- **High-cost beneficiary (classification)** (classification; target `high_cost`)

**Results from the test run**

- Medical charges (regression, raw scale): best model = Gradient Boosting (tuned)  (test R2 = 0.894, MAE = 2631.256, RMSE = 4412.863; mean-predictor RMSE = 13612.426)
  - Top features: smoker_x_bmi, age, age_x_bmi, bmi
- High-cost beneficiary (classification): best model = K-Nearest Neighbours (tuned)  (test F1 = 0.817, accuracy = 0.922, ROC-AUC = 0.878; majority-class baseline accuracy = 0.750)
  - Top features: bmi, age, smoker_x_bmi, age_x_bmi

**Discussion**

- On the raw cost scale the errors of smokers dominate RMSE; interaction features (smoker x BMI, age^2) help linear models, trees find them automatically.
- The high-cost classifier is accurate mainly because smoking and obesity identify most of the top quartile; recall of this group is the key figure for care management.
- Predictions are population-level estimates; they must not be used to deny coverage to individuals.

**Limitations and future work**

- Add diagnosis codes and prior-year utilisation for a realistic risk-adjustment model.
- Evaluate fairness across regions and sexes.

**Conclusion.** Smoking, age and BMI explain the bulk of the cost variation; the regression and the high-cost classifier give actionable, interpretable risk estimates.

---

### Project 16: Spotify tracks: genre classification and audio clustering

**Notebook:** `16_spotify_tracks_classification_clustering.ipynb`  
**Dataset:** Spotify Tracks / Spotify Songs (about 28,000 unique tracks, 6 genres)  
**Source:** TidyTuesday 2020-01-21 (Spotify API data); CSV: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv  
**Output folder:** `outputs/16_spotify/`

**About the data.** Spotify audio features (danceability, energy, loudness, valence, tempo ...), popularity, release date and playlist genre for about 32,000 track-playlist rows.

**Problem statement.** (1) Classify a track's playlist genre from its audio features. (2) Discover natural groups of tracks with clustering and compare them with the labelled genres.

**Objectives**

1. Remove duplicate tracks that appear in several playlists and clean missing text fields.
2. Explore audio features by genre.
3. Engineer release year, duration and interaction features.
4. Classification of 6 genres and K-Means / hierarchical / GMM / DBSCAN clustering with internal and external (ARI, NMI) evaluation.

**Modelling tasks**

- **Playlist genre (classification)** (classification; target `playlist_genre`)
- **Audio-feature clusters** (clustering; features `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`)

**Results from the test run**

- Playlist genre (classification): best model = Gradient Boosting (tuned)  (test F1 = 0.602, accuracy = 0.607, ROC-AUC = 0.878; majority-class baseline accuracy = 0.191)
  - Top features: release_year, tempo, speechiness, danceability
- Audio-feature clusters: best method = Agglomerative (Ward) with k = 2 (silhouette = 0.223, Davies-Bouldin = 1.637)

**Discussion**

- Accuracy is around 60% for six genres (chance is about 17-19%); the confusion matrix shows which genres are distinctive and which, typically pop, overlap with others. Release year is a surprisingly strong feature because genres differ in era.
- Genre labels come from playlists, not from the music, so a ceiling on achievable accuracy is expected.
- The unsupervised clusters barely align with the genre labels (ARI close to 0) and silhouette is low, so the audio space has no sharp natural groups; k = 2 mostly separates calm/acoustic from energetic tracks.

**Limitations and future work**

- Add artist or text features and audio embeddings for better genre separation.
- Use multi-label classification, because a track can belong to several genres.

**Conclusion.** Audio features carry real but limited information about genre (supervised accuracy well above chance), while unsupervised clusters are weak and cut across genre labels.

---

### Project 17: Netflix catalogue: EDA and content-based recommender

**Notebook:** `17_netflix_eda_recommendation.ipynb`  
**Dataset:** Netflix Movies and TV Shows (7,787 titles, to 2021)  
**Source:** Kaggle 'Netflix Movies and TV Shows' (Flixable); CSV copy: https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv  
**Output folder:** `outputs/17_netflix/`

**About the data.** Type, title, director, cast, country, date added, release year, rating, duration, genres and description of Netflix titles.

**Problem statement.** Understand how the Netflix catalogue is composed and has evolved, and build a content-based recommender that suggests similar titles.

**Objectives**

1. Clean missing director / cast / country values and parse dates and durations.
2. EDA: movies vs TV shows, growth over time, countries, genres, ratings, durations.
3. Feature engineering: genres, text tokens for cast / director / country, combined text.
4. Recommender with TF-IDF + cosine similarity; compare three feature sets by genre precision@10 against a random baseline; cluster titles with SVD + K-Means.

**Modelling tasks**

- **Custom pipeline** (not the standard supervised flow): content-based recommender using TF-IDF and cosine similarity, three feature sets compared by genre precision@10 against a random baseline, plus title clustering with TruncatedSVD and K-Means

**Results from the test run**

- Content-based recommender: best = TF-IDF cosine: Description + cast + director + country + rating with precision@10 = 0.520 versus random = 0.234
- Title clustering: best method = K-Means on TF-IDF SVD with k = 12 (silhouette = 0.066, Davies-Bouldin = 3.581)

**Discussion**

- The catalogue is dominated by movies, while the number of titles added per year grew sharply after 2015 (see the EDA plots).
- The United States and India lead production; international dramas and comedies are the largest genres.
- Adding cast, director, country and rating to the description text improves genre precision@10 from about 0.39 (description only) to about 0.52, and all variants beat random recommendation (about 0.23).
- Genre overlap is a proxy for relevance; real user feedback would be needed for a true evaluation.
- Title clusters are weak (silhouette about 0.07): they act as loose thematic groups rather than sharp categories.

**Limitations and future work**

- Use embeddings (e.g. sentence transformers) instead of TF-IDF.
- Combine with collaborative filtering once viewing data is available.

**Conclusion.** The EDA documents the strategic shift of Netflix towards TV and international content, and the content-based recommender returns related titles without any user data.

---

### Project 18: Mushroom edibility classification

**Notebook:** `18_mushroom_classification.ipynb`  
**Dataset:** Mushroom (8,124 records, 22 categorical attributes)  
**Source:** UCI Mushroom data set (Audubon Society Field Guide); CSV copy: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/mushrooms.csv  
**Output folder:** `outputs/18_mushroom/`

**About the data.** Physical characteristics of gilled mushrooms (cap, gills, stalk, odour, spore print, habitat ...) labelled edible or poisonous.

**Problem statement.** Decide from observable characteristics whether a mushroom is edible or poisonous. A false 'edible' prediction can be fatal, so recall of the poisonous class is critical.

**Objectives**

1. Clean the categorical data ('?' in stalk-root treated as its own category, constant columns removed).
2. Measure the association of every attribute with edibility (Cramer's V).
3. Create simple expert features (odour flags) and one-hot encode.
4. Compare classifiers; examine errors on the poisonous class.

**Modelling tasks**

- **Edibility** (classification; target `type`)

**Results from the test run**

- Edibility: best model = Decision Tree  (test F1 = 1.000, accuracy = 1.000, ROC-AUC = 1.000; majority-class baseline accuracy = 0.518)
  - Top features: odor_foul_group, gill_narrow, gill_size, spore_print_color

**Discussion**

- Odour is the most informative attribute: foul, fishy or spicy smells are almost always poisonous while 'none' is mostly edible; spore-print colour and gill size follow.
- Because the attributes are almost deterministic, tree-based models reach (nearly) perfect accuracy; this is a property of the dataset, not proof that mushroom foraging is solved.
- Even a tiny false-negative rate is unacceptable in practice, so the model must not replace expert identification.

**Limitations and future work**

- Real-world mushrooms include many species not in this data; test on out-of-distribution species.
- Combine with image classification for field use.

**Conclusion.** A handful of visible attributes, above all odour, separate poisonous from edible mushrooms almost perfectly in this dataset; interpretable decision rules are enough.

---

### Project 19: Dry bean variety classification

**Notebook:** `19_dry_bean_classification.ipynb`  
**Dataset:** Dry Bean Dataset (13,611 beans, 7 varieties, 16 shape features)  
**Source:** Koklu & Ozkan (2020), UCI Machine Learning Repository; Excel copy: https://raw.githubusercontent.com/modelMaestroSolo/Dry_Beans_Classifier/main/Dry_Bean_Dataset.xlsx  
**Output folder:** `outputs/19_dry_bean/`

**About the data.** Geometric measurements (area, perimeter, axes, eccentricity, roundness, shape factors) extracted from images of dry bean grains of seven registered varieties.

**Problem statement.** Automatically classify the variety of a dry bean from image-derived shape measurements, for uniform seed sorting.

**Objectives**

1. Remove duplicate rows and treat outliers in the shape measurements.
2. Show how strongly the geometric features are correlated and remove redundancy.
3. Create ratio features; scale for distance-based models.
4. Compare classifiers; inspect which varieties are confused.

**Modelling tasks**

- **Bean variety** (classification; target `Class`)

**Results from the test run**

- Bean variety: best model = Gradient Boosting (tuned)  (test F1 = 0.918, accuracy = 0.918, ROC-AUC = 0.993; majority-class baseline accuracy = 0.262)
  - Top features: Area, AspectRation, MinorAxisLength, roundness

**Discussion**

- Area, perimeter, convex area and equivalent diameter are almost perfectly correlated, so a few of them are removed without information loss.
- Bombay beans (much larger) are identified perfectly, while Sira (F1 about 0.86) is the hardest variety, mostly confused with Dermason; Barbunya and Cali also overlap.
- The best model reaches about 92% accuracy, in line with the 93% reported in the original paper.

**Limitations and future work**

- Colour and texture features from the images could separate visually similar varieties.
- Outlier capping is applied to all features; evaluating without capping could be checked.

**Conclusion.** Shape features alone classify seven bean varieties with high accuracy; the model can support automated seed quality control.

---

### Project 20: Obesity level prediction

**Notebook:** `20_obesity_levels_classification.ipynb`  
**Dataset:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition (2,111 people)  
**Source:** UCI Machine Learning Repository, dataset 544 (Palechor & De la Hoz Manotas, 2019): https://archive.ics.uci.edu/dataset/544  
**Output folder:** `outputs/20_obesity/`

**About the data.** Eating habits, physical condition, age, height and weight of people from Mexico, Peru and Colombia, with a 7-level obesity label (77% of rows were generated synthetically with SMOTE).

**Problem statement.** Predict a person's obesity level from lifestyle and body measurements, and test how much can be predicted from lifestyle alone.

**Objectives**

1. Clean the data (duplicates, outliers) and encode the survey answers.
2. Create BMI and lifestyle score features.
3. Task A: classify with all features. Task B: classify from lifestyle/demographics only (no height, weight or BMI) to avoid label leakage.
4. Compare models and discuss the synthetic-data caveat.

**Modelling tasks**

- **Obesity level (all features)** (classification; target `NObeyesdad`)
- **Obesity level (lifestyle only)** (classification; target `NObeyesdad`; excludes `Height`, `Weight`, `BMI`)

**Results.** Not listed here: this notebook was verified end to end on a stand-in file with the same columns because the UCI server was unreachable from the test environment. Run it on the real dataset to get your own numbers.

**Discussion**

- The label is defined from BMI, so with height, weight and BMI the task is almost deterministic and accuracy is very high: this shows consistency, not real predictive insight.
- Without anthropometrics (task B) accuracy is expected to drop clearly; family history of overweight, high-calorie food, snacking, transport and physical activity remain informative.
- 77% of rows are synthetic (SMOTE), so cross-validated accuracy is optimistic because synthetic neighbours of a test row can sit in the training set.

**Limitations and future work**

- Evaluate only on the 23% real records.
- Collect larger real survey samples and use ordinal classification, since the 7 levels are ordered.

**Conclusion.** Obesity level is predicted almost perfectly when body measurements are used; from lifestyle alone the model gives moderate accuracy and points to family history, diet and activity as modifiable factors.

---

## Output files

Each notebook creates `outputs/<project_slug>/` next to the notebook and saves:

| File | Content |
|------|---------|
| `raw_data_snapshot.csv` | Data as downloaded |
| `clean_data.csv` | After cleaning |
| `engineered_data.csv` | After feature engineering |
| `*.png` | Every figure (outliers, EDA, feature importance, confusion matrix, ROC, residuals, clusters) |
| `model_comparison_*.csv` | Metrics for every model, per task |
| `results_summary.csv` | Best model and key metrics per task |

Figures also appear inline in the notebook.

---

## Adapting a notebook to your own dataset

Everything dataset-specific lives in the **Configuration and hooks** cell, so the rest of the notebook can stay untouched.

1. Change `CONFIG["urls"]` to your CSV link (or set the `DATA_PATH` environment variable for a local file).
2. Update `title`, `problem`, `objectives`, `columns_info`, `discussion` and `conclusion`.
3. Set `drop_cols` (identifiers) and `force_categorical` (numeric codes that are really categories).
4. Define `tasks`, for example `dict(kind="classification", target="Survived")`, `dict(kind="regression", target="price", log_target=True)` or `dict(kind="clustering", features=[...], kmax=8)`.
5. Optionally edit `create_features(df)` to add your own columns.
6. Run all cells.

Optional hooks the notebooks look for by name: `read_source`, `custom_load`, `fallback_load`, `prepare_raw`, `eda_extra`, `create_features`, `custom_model_development`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Download fails or HTTP 403/404 | The dataset URL moved or your network blocks it. Download the file manually and set `DATA_PATH`, or in Colab use the upload dialog. |
| `DATA_PATH` usage | In a notebook cell run `import os; os.environ["DATA_PATH"] = "/path/to/file.csv"` before the load step. |
| Project 20 (Obesity) cannot download | UCI sometimes blocks automated downloads. Get `ObesityDataSet_raw_and_data_sinthetic.csv` from [archive.ics.uci.edu/dataset/544](https://archive.ics.uci.edu/dataset/544) and use `DATA_PATH`. |
| Notebook is very slow | Reduce `n_iter` in the `RandomizedSearchCV` call, or remove the slowest models from `get_models`. |
| `ModuleNotFoundError` | Run the `pip install` line from the quick start in the same environment as the kernel. |
| Plots do not appear | Make sure the notebook runs in Jupyter or Colab; add `%matplotlib inline` at the top if needed. |

---

## Notes on results

- The numbers shown for each project come from one verified run with `random_state = 42`. Small differences are normal across scikit-learn versions or if the dataset mirror changes.
- Test-set metrics on small datasets (Iris, Heart disease, Mall customers, Used cars, Car price) move by several points with a different split. Treat them as indicative, not exact.
- Several datasets contain known quirks that the discussion sections call out, for example the synthetic rows in the Obesity data, label leakage if BMI is used to predict the obesity level, and the near-perfect separability of the Mushroom data.
- These notebooks are for learning and portfolio use. They are not validated for medical, financial or other real-world decisions.
