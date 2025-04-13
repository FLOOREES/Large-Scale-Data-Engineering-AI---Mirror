# data_analysis_pipeline.py

from pathlib import Path
from typing import List
import traceback
import datetime
import joblib #type: ignore

# --- Core Imports ---
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

# --- ML Imports ---
from sklearn.preprocessing import StandardScaler    #type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error #type: ignore
import lightgbm as lgb #type: ignore

# --- Visualization ---
import matplotlib.pyplot as plt #type: ignore

class DataAnalysisPipeline:
    """
    A pipeline to perform data analysis and train a LightGBM model
    to predict average monthly rent using the consolidated exploitation data.
    ITERATION 4: Verification for data leakage.
    """

    def __init__(self, spark: SparkSession,
                 input_path: str = "./data/exploitation",
                 output_dir: str = "./data/analysis/model", # For plots/metrics
                 model_dir: str = "./models", # For model artifacts
                 target_variable: str = "avg_monthly_rent_eur"):
        # ... (init remains the same as your last version) ...
        self.spark = spark
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir) # Added model directory path
        self.target_variable = target_variable
        self.id_cols = ["municipality_id", "municipality_name", "comarca_name"]
        self.time_col = "any"
        self.numeric_features_base = [
            "total_annual_contracts", "salary_per_capita_eur", "population",
            "population_men", "population_women", "births_total",
            "population_spanish_nat", "surface_km2", "density_pop_km2",
            "longitude", "latitude", "unemployment_total_avg",
            "total_family_dwellings", "public_libraries_count",
            "sports_pavilions_count", "multi_sports_courts_count",
            "indoor_pools_count"
        ]
        self.categorical_features = ["comarca_name"]
        self.data_pd: pd.DataFrame = None
        self.features: List[str] = []
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.model: lgb.LGBMRegressor = None
        self.scaler: StandardScaler = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("Data Analysis Pipeline Initialized.")
        print(f"  Input Delta Path: {self.input_path}")
        print(f"  Analysis Output Directory: {self.output_dir}")
        print(f"  Model Output Directory: {self.model_dir}")
        print(f"  Target Variable: {self.target_variable}")


    # --- _load_data, _analyze_data remain unchanged ---
    def _load_data(self):
        """Loads data from the Exploitation Zone Delta table into Pandas."""
        print("\n--- Step 1: Loading Data ---")
        # ... (Keep identical) ...
        try:
            delta_log_path = Path(self.input_path) / "_delta_log"
            if not delta_log_path.is_dir():
                raise FileNotFoundError(f"Input Delta table log not found at {delta_log_path}.")
            print(f"Reading Delta table from: {self.input_path}")
            df_spark = self.spark.read.format("delta").load(self.input_path)
            print(f"Loaded {df_spark.count()} rows from Delta table.")
            print("Converting Spark DataFrame to Pandas DataFrame...")
            start_time = datetime.datetime.now()
            self.data_pd = df_spark.toPandas()
            end_time = datetime.datetime.now()
            print(f"Conversion to Pandas complete in {end_time - start_time}.")
            print(f"Pandas DataFrame shape: {self.data_pd.shape}")
            mem_usage = self.data_pd.memory_usage(deep=True).sum() / (1024**2)
            print(f"Pandas DataFrame memory usage: {mem_usage:.2f} MB")
            if mem_usage > 1000:
                 print("WARNING: Pandas DataFrame size is large.")
        except Exception as e:
            print(f"!!! ERROR during data loading: {e}")
            traceback.print_exc()
            raise

    def _analyze_data(self):
        """Performs basic EDA on the loaded Pandas DataFrame."""
        if self.data_pd is None: return
        print("\n--- Step 2: Basic Data Analysis ---")
        # ... (Keep identical) ...
        print("\nDataFrame Info:")
        self.data_pd.info()
        print("\nDescriptive Statistics (Numeric Columns):")
        numeric_cols_pd = self.data_pd.select_dtypes(include=np.number).columns.tolist()
        print(self.data_pd[numeric_cols_pd].describe().to_string())
        print("\nMissing Value Counts:")
        missing_counts = self.data_pd.isnull().sum()
        print(missing_counts[missing_counts > 0].to_string())
        print("\nBasic Data Analysis complete.")

    # --- _feature_engineering remains unchanged ---
    def _feature_engineering(self):
        """Creates new features: lags, differences, rolling means."""
        if self.data_pd is None: return
        print("\n--- Step 3: Feature Engineering ---")
        print("Sorting data by municipality and year...")
        self.data_pd = self.data_pd.sort_values(by=[*self.id_cols, self.time_col]).reset_index(drop=True)
        print("Creating time-based features (lags, diffs, rolling)...")
        target = self.target_variable
        salary = "salary_per_capita_eur"
        contracts = "total_annual_contracts"
        engineered_features = []

        # Lags (Keep as before)
        for col in [target, salary, contracts]:
            if col in self.data_pd.columns:
                for lag in [1, 2]:
                    lag_col_name = f"{col}_lag{lag}"
                    self.data_pd[lag_col_name] = self.data_pd.groupby(self.id_cols)[col].shift(lag)
                    engineered_features.append(lag_col_name)

        # Differences (CORRECTED: Use lag1 - lag2)
        lag1_target_col = f"{target}_lag1"
        lag2_target_col = f"{target}_lag2"
        if lag1_target_col in self.data_pd.columns and lag2_target_col in self.data_pd.columns:
             diff_col_name = f"{target}_diff1" # Name still represents 1-period difference
             # Calculate difference between T-1 and T-2
             self.data_pd[diff_col_name] = self.data_pd[lag1_target_col] - self.data_pd[lag2_target_col]
             engineered_features.append(diff_col_name)
             print(f"Created corrected difference feature: {diff_col_name} (lag1 - lag2)")
        else:
             print(f"Skipping difference feature creation as lag1 or lag2 target column is missing.")


        # Rolling Means (Keep as before - based on lag1)
        if lag1_target_col in self.data_pd.columns:
            roll_col_name = f"{target}_roll_mean3"
            self.data_pd[roll_col_name] = self.data_pd.groupby(self.id_cols)[lag1_target_col] \
                                              .rolling(window=3, min_periods=1).mean().reset_index(level=self.id_cols, drop=True)
            engineered_features.append(roll_col_name)

        # --- Define final feature list ---
        self.features = self.numeric_features_base + [self.time_col] + engineered_features + self.categorical_features
        self.features = [f for f in self.features if f in self.data_pd.columns] # Ensure only existing columns are included
        print(f"Engineered features created. Final candidate features ({len(self.features)}): {self.features}")

        # --- Handle NaNs introduced by essential lags/target ---
        essential_cols_for_dropna = [self.target_variable, lag1_target_col] # Still drop if target or lag1 is missing
        print(f"Dropping rows only if NaNs exist in essential columns: {essential_cols_for_dropna}")
        initial_rows = len(self.data_pd)
        # Also drop rows where the corrected diff1 could not be calculated (because lag2 was missing)
        # If diff_col_name was created, add it to the drop subset
        if 'diff_col_name' in locals() and diff_col_name in self.data_pd.columns:
             essential_cols_for_dropna.append(diff_col_name)
             print(f"Also dropping rows if corrected '{diff_col_name}' is NaN.")

        self.data_pd = self.data_pd.dropna(subset=essential_cols_for_dropna).reset_index(drop=True)
        rows_after_drop = len(self.data_pd)
        print(f"Dropped {initial_rows - rows_after_drop} rows based on NaNs in essential columns.")
        print(f"DataFrame shape after essential drop: {self.data_pd.shape}")

        print("\nMissing values REMAINING in candidate features (to be handled by LightGBM):")
        remaining_missing = self.data_pd[self.features].isnull().sum()
        print(remaining_missing[remaining_missing > 0].to_string())

        print("Feature Engineering complete.")

    # --- Preprocessing: ADDED Verification Prints ---
    def _preprocess_data(self):
        """Splits data, handles categoricals, and scales features."""
        if self.data_pd is None or not self.features: return
        print("\n--- Step 4: Preprocessing Data ---")

        # --- 4a. Prepare Categorical Features BEFORE Split ---
        print("Preparing categorical features for LightGBM (converting dtype)...")
        self.categorical_features_in_X = [f for f in self.features if f in self.categorical_features and f in self.data_pd.columns]
        if self.categorical_features_in_X:
            print(f"Converting columns {self.categorical_features_in_X} to 'category' dtype in main DataFrame...")
            for col in self.categorical_features_in_X:
                # Convert in the main DataFrame before any splits
                self.data_pd[col] = self.data_pd[col].astype('category')
            print("Categorical dtypes set.")
            print("dtypes after conversion:", self.data_pd[self.categorical_features_in_X].dtypes)
        else:
            print("No specified categorical features found in the feature list.")
            self.categorical_features_in_X = [] # Ensure list is empty

        # --- 4b. Split Data (Time-Based) ---
        print("\nSplitting data into training and testing sets (time-based)...")
        test_year = self.data_pd[self.time_col].max()
        print(f"Test year determined as: {test_year}")
        # Use .copy() to explicitly create copies
        train_df = self.data_pd[self.data_pd[self.time_col] < test_year].copy()
        test_df = self.data_pd[self.data_pd[self.time_col] == test_year].copy()
        if len(train_df) == 0 or len(test_df) == 0:
             raise ValueError(f"Train or test set is empty after splitting on year {test_year}.")
        # Verification Prints
        print(f"VERIFICATION: Training data year range: {train_df[self.time_col].min()} - {train_df[self.time_col].max()}")
        print(f"VERIFICATION: Test data year range: {test_df[self.time_col].min()} - {test_df[self.time_col].max()}")
        train_years = set(train_df[self.time_col].unique())
        test_years = set(test_df[self.time_col].unique())
        if train_years.intersection(test_years):
             print("!!! WARNING: Train and Test sets have overlapping years! Check split logic. !!!")
        # Assign features to X_train/X_test AFTER the split
        self.X_train = train_df[self.features]
        self.y_train = train_df[self.target_variable]
        self.X_test = test_df[self.features]
        self.y_test = test_df[self.target_variable]
        print(f"Data split: Train shape={self.X_train.shape}, Test shape={self.X_test.shape}")

        # --- 4c. Verify Test Set Categories (Optional but Recommended) ---
        if self.categorical_features_in_X:
            print("Verifying test set categories match training set...")
            for col in self.categorical_features_in_X:
                if col in self.X_train.columns and col in self.X_test.columns:
                    train_cats = self.X_train[col].cat.categories
                    test_cats = self.X_test[col].cat.categories
                    if not train_cats.equals(test_cats):
                        print(f"WARNING: Categories for '{col}' differ between train and test AFTER split.")
                        # This *shouldn't* happen if split correctly, but good to check.
                        # If it does, we might need to re-apply categories like before:
                        # self.X_test.loc[:, col] = pd.Categorical(self.X_test[col], categories=train_cats)
                else:
                    print(f"Warning: Column '{col}' missing from train or test during category verification.")
            print("Test set category check complete (no explicit re-application done here).")


        # --- 4d. Missing Values ---
        print("\nSkipping explicit missing value imputation.")


        # --- 4e. Scale Numeric Features ---
        print("\nScaling numeric features...")
        numeric_features_to_scale = self.X_train.select_dtypes(include=np.number).columns.tolist()
        if self.time_col in numeric_features_to_scale:
            print(f"NOTE: Temporarily removing '{self.time_col}' from scaling list.")
            numeric_features_to_scale.remove(self.time_col)

        if not numeric_features_to_scale:
             print("No numeric features found to scale (excluding year).")
        else:
            self.scaler = StandardScaler()
            print(f"Fitting scaler on training data (features: {numeric_features_to_scale})...")
            self.scaler.fit(self.X_train[numeric_features_to_scale])

            print("Transforming training and testing data...")
            X_train_scaled_np = self.scaler.transform(self.X_train[numeric_features_to_scale])
            X_test_scaled_np = self.scaler.transform(self.X_test[numeric_features_to_scale])
            X_train_scaled_df = pd.DataFrame(X_train_scaled_np, index=self.X_train.index, columns=numeric_features_to_scale)
            X_test_scaled_df = pd.DataFrame(X_test_scaled_np, index=self.X_test.index, columns=numeric_features_to_scale)
            self.X_train.loc[:, numeric_features_to_scale] = X_train_scaled_df
            self.X_test.loc[:, numeric_features_to_scale] = X_test_scaled_df
            scaler_path = self.model_dir / 'standard_scaler.joblib'
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
            print("Feature scaling complete.")

        print("Preprocessing complete.")

    # --- Model Training: ADDED Verification Prints ---
    def _train_model(self):
        """
        Trains the LightGBM model with early stopping on a validation set,
        then retrains on the full training set using the best iteration found.
        """
        if self.X_train is None or self.y_train is None: return
        print("\n--- Step 5: Training LightGBM Model ---")

        # --- Prepare for Early Stopping ---
        val_year = self.X_train[self.time_col].max()
        train_final_idx = self.X_train[self.X_train[self.time_col] < val_year].index
        val_idx = self.X_train[self.X_train[self.time_col] == val_year].index

        X_train_for_stopping = self.X_train.loc[train_final_idx]
        y_train_for_stopping = self.y_train.loc[train_final_idx]
        X_val = self.X_train.loc[val_idx]
        y_val = self.y_train.loc[val_idx]

        callbacks = []
        best_iteration = None # Initialize variable to store best iteration
        if len(X_val) == 0:
             print("Warning: No data for validation year. Training without early stopping.")
             eval_set = None
             # Train directly on full training data if no validation set
             X_train_final = self.X_train
             y_train_final = self.y_train
             n_estimators_final = 2000 # Use original large number if no early stopping
        else:
             print(f"Using year {int(X_val[self.time_col].iloc[0])} as validation set for early stopping.")
             print(f"Initial training shape: {X_train_for_stopping.shape}, Validation shape: {X_val.shape}")
             eval_set = [(X_val, y_val)]
             early_stopping_callback = lgb.early_stopping(stopping_rounds=50, verbose=True)
             callbacks.append(early_stopping_callback)
             # Use the split data ONLY for finding the best iteration
             X_train_final = X_train_for_stopping
             y_train_final = y_train_for_stopping
             n_estimators_final = 2000 # Start with high number for early stopping run

        # --- LightGBM Parameters ---
        params = {
            'objective': 'regression_l1', 'metric': ['mae', 'rmse'],
            # 'n_estimators': n_estimators_final, # Set n_estimators dynamically later
            'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 1, 'num_leaves': 41, 'max_depth': -1, 'lambda_l1': 0.1,
            'lambda_l2': 0.1, 'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt'
        }

        print(f"\nStep 5a: Finding best iteration using validation set (if available)...")
        temp_model = lgb.LGBMRegressor(**params, n_estimators=n_estimators_final) # Use high n_estimators here

        start_time_fit1 = datetime.datetime.now()
        categorical_features_in_X = [f for f in self.features if f in self.categorical_features]
        temp_model.fit(X_train_final, y_train_final, # Fit on training data up to val year
                       eval_set=eval_set,
                       eval_metric=['mae', 'rmse'],
                       callbacks=callbacks if eval_set else None,
                       categorical_feature=self.categorical_features_in_X if self.categorical_features_in_X else 'auto'
                      )
        end_time_fit1 = datetime.datetime.now()
        print(f"Initial fit complete in {end_time_fit1 - start_time_fit1}.")

        # Get the best iteration if early stopping was used
        if eval_set and hasattr(temp_model, 'best_iteration_') and temp_model.best_iteration_ is not None:
            best_iteration = temp_model.best_iteration_
            print(f"Best iteration found via early stopping: {best_iteration}")
            # Update n_estimators in params for the final model
            params['n_estimators'] = best_iteration
        else:
            print("No early stopping performed or best iteration not found. Using original n_estimators.")
            params['n_estimators'] = n_estimators_final # Use the original large number


        # --- Step 5b: Retrain on FULL training data using best iteration ---
        test_year = self.X_test[self.time_col].max()
        print(f"\nStep 5b: Retraining final model on full training set (years < {test_year}) for {params['n_estimators']} iterations...")
        # Use the FULL self.X_train and self.y_train here
        self.model = lgb.LGBMRegressor(**params) # Create new model instance with optimal n_estimators
        start_time_fit2 = datetime.datetime.now()
        self.model.fit(self.X_train, self.y_train, # Fit on ALL training data
                       categorical_feature=self.categorical_features_in_X if self.categorical_features_in_X else 'auto'
                      )
        end_time_fit2 = datetime.datetime.now()
        print(f"Final model retraining complete in {end_time_fit2 - start_time_fit2}.")

        # --- Save the FINAL retrained model ---
        model_path = self.model_dir / 'rent_predictor_lgbm.joblib'
        joblib.dump(self.model, model_path)
        print(f"Final retrained model saved to {model_path}")

    # --- Evaluation: ADDED Verification Print ---
    def _evaluate_model(self):
        """Makes predictions, evaluates metrics, and generates plots."""
        if self.model is None or self.X_test is None or self.y_test is None: return
        print("\n--- Step 6: Evaluating Model ---")

        # *** Verification Print 3: Sample of Test Features ***
        print(f"VERIFICATION: Sample of features used for prediction on test set (Year {self.X_test[self.time_col].iloc[0]}):")
        # Show year, lag1 target, lag1 salary (demonstrates info available)
        cols_to_show = [self.time_col, f"{self.target_variable}_lag1", "salary_per_capita_eur_lag1"]
        cols_to_show = [c for c in cols_to_show if c in self.X_test.columns] # Ensure columns exist
        print(self.X_test[cols_to_show].head(5).to_string())
        # *** End Verification Print 3 ***

        print("\nMaking predictions on the test set...")
        y_pred = self.model.predict(self.X_test)

        # --- Calculate Metrics ---
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        print("\nEvaluation Metrics on Test Set:")
        print(f"  Mean Absolute Error (MAE):  {mae:.2f}")
        print(f"  Mean Squared Error (MSE):   {mse:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2%}")
        print(f"  R-squared (R²):           {r2:.4f}")

        # --- Plots (Keep as before) ---
        # ... (Feature Importance, Pred vs Actual, Residual plots) ...
        # --- Feature Importance Plot ---
        try:
            print("\nGenerating Feature Importance plot...")
            lgb.plot_importance(self.model, max_num_features=20, figsize=(10, 8), importance_type='gain')
            plt.title('LightGBM Feature Importance (Top 20 by Gain)')
            plt.tight_layout()
            plot_path = self.output_dir / 'feature_importance.png'
            plt.savefig(plot_path)
            print(f"Feature importance plot saved to {plot_path}")
            plt.close()
        except Exception as fi_e:
            print(f"Could not generate feature importance plot: {fi_e}")
        # --- Prediction vs Actual Plot ---
        try:
            print("\nGenerating Prediction vs Actual plot...")
            plt.figure(figsize=(8, 8))
            plt.scatter(self.y_test, y_pred, alpha=0.5, label='Predictions')
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
            plt.xlabel("Actual Rent (€)")
            plt.ylabel("Predicted Rent (€)")
            plt.title(f"Prediction vs Actual Rent (Test Set, R²={r2:.3f})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plot_path = self.output_dir / 'prediction_vs_actual.png'
            plt.savefig(plot_path)
            print(f"Prediction vs Actual plot saved to {plot_path}")
            plt.close()
        except Exception as pvsa_e:
             print(f"Could not generate Prediction vs Actual plot: {pvsa_e}")
        # --- Residual Plot ---
        try:
            print("\nGenerating Residual plot...")
            residuals = self.y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.xlabel("Predicted Rent (€)")
            plt.ylabel("Residuals (Actual - Predicted)")
            plt.title("Residuals vs Predicted Rent (Test Set)")
            plt.grid(True)
            plt.tight_layout()
            plot_path = self.output_dir / 'residuals_vs_predicted.png'
            plt.savefig(plot_path)
            print(f"Residual plot saved to {plot_path}")
            plt.close()
        except Exception as res_e:
             print(f"Could not generate Residual plot: {res_e}")

    # --- run method remains unchanged ---
    def run(self):
        """Runs the full data analysis and modeling pipeline."""
        print("==================================================")
        print("=== Starting Data Analysis and Prediction Pipeline ===")
        print("==================================================")
        # ... (Keep identical) ...
        try:
            self._load_data()
            self._analyze_data()
            self._feature_engineering()
            self._preprocess_data()
            self._train_model()
            self._evaluate_model()
            print("\n==================================================")
            print("=== Pipeline Execution Successfully Completed ====")
            print("==================================================")
        except Exception as e:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!! Pipeline Execution Failed: {e}")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    from src.spark_session import get_spark_session
    spark = None
    try:
        spark = get_spark_session()
        pipeline = DataAnalysisPipeline(spark=spark)
        pipeline.run()

    except Exception as main_error:
        print(f"\nAn unexpected error occurred in the main execution block: {main_error}")
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping Spark Session.")
            spark.stop()
            print("Spark Session stopped.")