Problem Statement
The existing traditional ML methods notebook for sleep apnea detection (DREAMT PPG dataset) contains multiple correctness bugs, design contradictions, and missing infrastructure that prevent it from producing valid, comparable experimental results. Specifically: SVC is referenced but never imported; RandomOverSampler is applied globally before CV folds, causing data leakage; class weights and oversampling are applied simultaneously and redundantly; hyperparameters are arbitrary placeholders; the primary metric (accuracy) is misleading on a 4.3:1 imbalanced test set; there is no structured comparison table; and the preprocessing pipeline does not include patient IDs, making patient-level CV impossible.

Solution
Refactor both the preprocessing notebook and the ML methods notebook to produce a clean, reproducible, clinically-grounded comparison of traditional ML models on the DREAMT PPG feature-extracted dataset. The refactored system will: fix all import and logic bugs; implement patient-level 5-fold CV with oversampling correctly inside each fold; use nested CV for hyperparameter tuning (inner loop) and threshold selection (outer loop); report sensitivity as the headline metric with a ≥80% clinical floor; and output a structured comparison table across all model × dimensionality reduction combinations.

User Stories
As a researcher, I want SVC correctly imported from sklearn.svm with probability=True, so that SVM pipelines run without NameError and support predict_proba for threshold tuning.
As a researcher, I want RandomOverSampler applied inside the imblearn pipeline rather than globally, so that oversampled data never leaks into CV validation folds.
As a researcher, I want class weights removed from all RandomForestClassifier instances, so that imbalance is handled solely by oversampling and the two strategies are not applied simultaneously.
As a researcher, I want the preprocessing notebook to emit a patient ID column in every output CSV, so that patient-level CV fold splitting is possible without reconstructing segment boundaries.
As a researcher, I want the conflicting TEST_IDS definitions in the preprocessing notebook resolved, so that the train/test split is unambiguous and reproducible.
As a researcher, I want a 5-fold patient-level CV outer loop, so that threshold selection never touches the held-out test patients.
As a researcher, I want a GridSearchCV inner loop scored on MCC, so that hyperparameters (KNN k, RF n_estimators, PCA n_components, SelectKBest k, SVM C/gamma) are chosen without using the test set.
As a researcher, I want the scaler to vary by dimensionality reduction method (MinMaxScaler for chi-square, StandardScaler for all others, no scaler for RF with no non-negative constraint), so that each pipeline satisfies the mathematical requirements of its components.
As a researcher, I want threshold tuning performed on held-out CV fold predictions via predict_proba, sweeping to find the operating point achieving ≥80% sensitivity, so that the decision threshold is calibrated without leaking test set information.
As a researcher, I want the median CV threshold applied to the final held-out test set exactly once, so that final metrics are unbiased.
As a researcher, I want sensitivity (apnea-class recall) as the headline metric with ≥80% marked as the clinical floor, so that clinically unacceptable models are immediately identifiable.
As a researcher, I want MCC reported as the single summary statistic, so that model quality across both classes is captured in a way that is robust to class imbalance.
As a researcher, I want a final comparison table showing sensitivity, MCC, specificity, precision, apnea-class F1, AUC-ROC, and CV-tuned threshold for each model × dimensionality reduction combination, so that results can be compared at a glance.
As a researcher, I want a naive baseline row ("always predict non-apnea": 0% sensitivity, 0.0 MCC, 81% accuracy) in the comparison table, so that the cost of the accuracy metric is visible.
As a researcher, I want all dead code cells (class balancing math, commented-out oversampling variants, "No Dim Reduction" sections) removed, so that the notebook is clean and unambiguous.
As a researcher, I want imblearn.pipeline.Pipeline used throughout instead of sklearn.pipeline.Pipeline, so that RandomOverSampler is a proper pipeline step.
As a researcher, I want the experiment matrix fixed at KNN × RF × SVM crossed with SelectKBest (f_classif), Chi-square, PCA, LDA — 12 combinations total — with no "no dim reduction" condition, so that the comparison is well-defined.
As a researcher, I want the model_pipeline helper refactored to build the step list dynamically (not via conditional branching), so that adding new steps doesn't require adding new if/else branches.
As a researcher, I want model_report extended to return predict_proba scores alongside confusion matrix metrics, so that AUC-ROC and threshold tuning have the data they need.
Implementation Decisions
Modules to build or modify
Preprocessing notebook (ENEE408N_preprocessing.ipynb):

Fix TEST_IDS — remove the duplicate conflicting definition; keep TEST_IDS = a[(a % 5 == 0) & (a != 60)] as the single canonical definition
Add a patient_id column to every row before writing any CSV, derived from the patient loop variable in save_ppg_to_csv_by_split
Ensure feature-extracted CSVs also carry the patient_id column through the feature_extraction_helper map step
ML methods notebook (ENEE408N_Traditional_ML_Methods.ipynb):

Imports: replace sklearn.pipeline.Pipeline with imblearn.pipeline.Pipeline; add from sklearn.svm import SVC; add from sklearn.metrics import roc_auc_score, matthews_corrcoef; add from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
Data loading: drop patient_id column into a separate groups array before creating X_train/y_train; remove global RandomOverSampler call
build_pipeline(model, scaler, dim_reduction, oversampler): dynamically construct step list; always include RandomOverSampler as a step so it runs inside CV folds
param_grid(model_name, dim_reduction_name): return a dict of hyperparameter search spaces keyed by pipeline step name, covering k (KNN), n_estimators/max_depth (RF), C/gamma (SVM), k (SelectKBest/chi2), n_components (PCA)
run_experiment(X, y, groups, model, param_grid, scaler, dim_reduction): outer StratifiedGroupKFold(n_splits=5) loop; inner GridSearchCV scored on MCC; collect predict_proba on each outer fold's held-out set; sweep thresholds to find ≥80% sensitivity operating point; record per-fold threshold; return best hyperparams, median threshold, and CV metrics
evaluate_on_test(pipeline, X_test, y_test, threshold): call predict_proba, apply threshold, compute sensitivity, MCC, specificity, precision, F1, AUC-ROC
Scaler selection logic: MinMaxScaler when dim reduction is chi-square; StandardScaler when model is KNN or SVM with any other dim reduction; None for RF
Results collection: accumulate one row per model × dim reduction combination into a results list; convert to DataFrame and display as final cell
Naive baseline row: hard-coded entry with 0% sensitivity, 0.0 MCC, 100% specificity, ~81% accuracy
Architectural decisions
Oversampler lives inside the imblearn pipeline so it is fit only on training folds, never validation folds
StratifiedGroupKFold is used so folds are stratified by label AND grouped by patient — no patient's segments span two folds
GridSearchCV inner loop uses a fixed random subset of training data if full training is prohibitively slow for SVM (noted as a tuning concern)
Test set is touched exactly once, after all CV decisions are finalized
predict_proba column index 1 corresponds to the apnea class (label=1); verified at runtime
Testing Decisions
A good test verifies external behavior: given known inputs, assert the output metrics and threshold are within expected bounds. Do not test sklearn/imblearn internals.
build_pipeline: given a model, scaler, dim_reduction, and oversampler, assert the returned pipeline has the correct named steps in the correct order
run_experiment: given a small synthetic balanced dataset with known patient groups, assert that (a) no patient appears in both train and validation of any fold, and (b) the returned threshold produces ≥80% sensitivity on the synthetic validation data
evaluate_on_test: given a fixed predict_proba output and ground truth, assert all metrics match hand-calculated values
No tests required for the preprocessing notebook (it is a data transformation script; correctness is verified by inspecting CSV output column headers)
Out of Scope
Deep learning / neural network models (mentioned as a future section in the notebook but not part of this refactor)
GPU-accelerated cuml SVM (commented out; not part of this refactor)
Automated model deployment or joblib export of all 12 trained models (save only the best-performing model)
Hyperparameter tuning for LDA (only one valid value: n_components=1 for binary classification)
Statistical significance testing across model comparisons
SHAP or feature importance analysis
Further Notes
The DREAMT dataset is loaded from HuggingFace (bsaenz/dreamt); patient IDs are integers in the range 2–103, with every 5th patient (excluding 60) reserved for test
The 4.3:1 class imbalance in the test set (81% non-apnea / 19% apnea) is intentional and reflects real-world conditions; the training set is balanced via oversampling
The ≥80% sensitivity clinical floor is benchmarked against FDA-cleared home sleep testing devices; Apple Watch's 66.3% sensitivity is explicitly not the target (that device is a notification tool, not a screener)
MCC is preferred over F1 as summary statistic per Chicco & Jurman (2020), because F1 ignores true negatives and can appear favorable for models with high false alarm rates