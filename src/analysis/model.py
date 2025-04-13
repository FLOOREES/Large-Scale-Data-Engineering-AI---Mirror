# data_analysis_pipeline.py

from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback
import datetime
import joblib #type: ignore

# --- Core Imports ---
import pandas as pd
import numpy as np

# --- PySpark Imports ---
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# --- ML Imports ---
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.preprocessing import StandardScaler    #type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error #type: ignore
import lightgbm as lgb #type: ignore

# --- Visualization ---
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore

# --- Delta Lake Package Configuration ---
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0"

class DataAnalysisPipeline:
    """
    A pipeline to perform data analysis and train a LightGBM model
    to predict average monthly rent using the consolidated exploitation data.
    ITERATION 4: Verification for data leakage.
    """

    def __init__(self, spark: SparkSession,
                 input_path: str = "./data/exploitation/consolidated_municipal_annual",
                 output_dir: str = "./data/analysis_outputs", # For plots/metrics
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
        # ... (Keep identical) ...
        print("Sorting data by municipality and year...")
        self.data_pd = self.data_pd.sort_values(by=[*self.id_cols, self.time_col]).reset_index(drop=True)
        print("Creating time-based features (lags, diffs, rolling)...")
        target = self.target_variable
        salary = "salary_per_capita_eur"
        contracts = "total_annual_contracts"
        engineered_features = []
        for col in [target, salary, contracts]:
            if col in self.data_pd.columns:
                for lag in [1, 2]:
                    lag_col_name = f"{col}_lag{lag}"
                    self.data_pd[lag_col_name] = self.data_pd.groupby(self.id_cols)[col].shift(lag)
                    engineered_features.append(lag_col_name)
        if f"{target}_lag1" in self.data_pd.columns:
             diff_col_name = f"{target}_diff1"
             self.data_pd[diff_col_name] = self.data_pd[target] - self.data_pd[f"{target}_lag1"]
             engineered_features.append(diff_col_name)
        if f"{target}_lag1" in self.data_pd.columns:
            roll_col_name = f"{target}_roll_mean3"
            self.data_pd[roll_col_name] = self.data_pd.groupby(self.id_cols)[f"{target}_lag1"] \
                                              .rolling(window=3, min_periods=1).mean().reset_index(level=self.id_cols, drop=True)
            engineered_features.append(roll_col_name)
        self.features = self.numeric_features_base + [self.time_col] + engineered_features + self.categorical_features
        self.features = [f for f in self.features if f in self.data_pd.columns]
        print(f"Engineered features created. Final candidate features ({len(self.features)}): {self.features}")
        essential_cols_for_dropna = [self.target_variable, f"{target}_lag1"]
        print(f"Dropping rows only if NaNs exist in essential columns: {essential_cols_for_dropna}")
        initial_rows = len(self.data_pd)
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

        # --- 4a. Split Data (Time-Based) ---
        print("Splitting data into training and testing sets (time-based)...")
        test_year = self.data_pd[self.time_col].max()
        print(f"Test year determined as: {test_year}")

        train_df = self.data_pd[self.data_pd[self.time_col] < test_year].copy()
        test_df = self.data_pd[self.data_pd[self.time_col] == test_year].copy()
        if len(train_df) == 0 or len(test_df) == 0:
             raise ValueError(f"Train or test set is empty after splitting on year {test_year}.")

        # *** Verification Print 1: Train/Test Year Ranges ***
        print(f"VERIFICATION: Training data year range: {train_df[self.time_col].min()} - {train_df[self.time_col].max()}")
        print(f"VERIFICATION: Test data year range: {test_df[self.time_col].min()} - {test_df[self.time_col].max()}")
        # Check for overlap
        train_years = set(train_df[self.time_col].unique())
        test_years = set(test_df[self.time_col].unique())
        if train_years.intersection(test_years):
             print("!!! WARNING: Train and Test sets have overlapping years! Check split logic. !!!")
        # *** End Verification Print 1 ***

        self.X_train = train_df[self.features]
        self.y_train = train_df[self.target_variable]
        self.X_test = test_df[self.features]
        self.y_test = test_df[self.target_variable]
        print(f"Data split: Train shape={self.X_train.shape}, Test shape={self.X_test.shape}")

        # --- 4b. Missing Values ---
        print("Skipping explicit missing value imputation.")

        # --- 4c. Categorical Features ---
        print("Preparing categorical features for LightGBM...")
        # ... (Keep identical) ...
        categorical_features_in_X = [f for f in self.features if f in self.categorical_features]
        if categorical_features_in_X:
            print(f"Converting categorical columns {categorical_features_in_X} to pandas 'category' dtype...")
            for col in categorical_features_in_X:
                 self.X_train[col] = self.X_train[col].astype('category')
                 self.X_test[col] = pd.Categorical(self.X_test[col], categories=self.X_train[col].cat.categories)
            print("Categorical features prepared.")
        else:
            print("No specified categorical features found in the feature list.")


        # --- 4d. Scale Numeric Features ---
        print("Scaling numeric features...")
        # ... (Keep identical logic including .loc fix and scaler saving) ...
        numeric_features_to_scale = self.X_train.select_dtypes(include=np.number).columns.tolist()
        if not numeric_features_to_scale:
             print("No numeric features found to scale.")
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
        """Trains the LightGBM model with early stopping."""
        if self.X_train is None or self.y_train is None: return
        print("\n--- Step 5: Training LightGBM Model ---")

        # --- Prepare for Early Stopping ---
        val_year = self.X_train[self.time_col].max()
        train_final_idx = self.X_train[self.X_train[self.time_col] < val_year].index
        val_idx = self.X_train[self.X_train[self.time_col] == val_year].index
        X_train_final = self.X_train.loc[train_final_idx]
        y_train_final = self.y_train.loc[train_final_idx]
        X_val = self.X_train.loc[val_idx]
        y_val = self.y_train.loc[val_idx]

        callbacks = []
        if len(X_val) == 0:
             print("Warning: No data for validation year. Training without early stopping.")
             eval_set = None
             # *** Verification Print 2a: Training Year Range (No Validation) ***
             print(f"VERIFICATION: Final training data year range: {X_train_final[self.time_col].min()} - {X_train_final[self.time_col].max()}")
        else:
             val_actual_year = int(X_val[self.time_col].iloc[0])
             print(f"Using year {val_actual_year} as validation set.")
             print(f"Final training shape: {X_train_final.shape}, Validation shape: {X_val.shape}")
             # *** Verification Print 2b: Train/Validation Year Ranges ***
             print(f"VERIFICATION: Final training data year range: {X_train_final[self.time_col].min()} - {X_train_final[self.time_col].max()}")
             print(f"VERIFICATION: Validation data year range: {X_val[self.time_col].min()} - {X_val[self.time_col].max()}")
             # Check overlap
             train_final_years = set(X_train_final[self.time_col].unique())
             val_years = set(X_val[self.time_col].unique())
             if train_final_years.intersection(val_years):
                  print("!!! WARNING: Final Train and Validation sets have overlapping years! Check split logic. !!!")
             # *** End Verification Print 2b ***
             eval_set = [(X_val, y_val)]
             early_stopping_callback = lgb.early_stopping(stopping_rounds=50, verbose=True)
             callbacks.append(early_stopping_callback)

        # --- LightGBM Parameters (Unchanged) ---
        params = {
            'objective': 'regression_l1', 'metric': ['mae', 'rmse'], 'n_estimators': 2000,
            'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 1, 'num_leaves': 41, 'max_depth': -1, 'lambda_l1': 0.1,
            'lambda_l2': 0.1, 'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt'
        }
        print(f"Training LGBMRegressor with parameters: {params}")
        self.model = lgb.LGBMRegressor(**params)
        start_time = datetime.datetime.now()
        categorical_features_in_X = [f for f in self.features if f in self.categorical_features]
        self.model.fit(X_train_final, y_train_final, eval_set=eval_set,
                       eval_metric=['mae', 'rmse'], callbacks=callbacks if eval_set else None,
                       categorical_feature=categorical_features_in_X if categorical_features_in_X else 'auto')
        end_time = datetime.datetime.now()
        print(f"Model training complete in {end_time - start_time}.")

        # --- Save Model ---
        model_path = self.model_dir / 'rent_predictor_lgbm.joblib'
        joblib.dump(self.model, model_path)
        print(f"Trained model saved to {model_path}")

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

# --- Spark Session Creation Helper (Unchanged) ---
# ... (Keep identical) ...
def get_spark_session() -> SparkSession:
    """Initializes and returns a SparkSession configured for Delta Lake."""
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("DataAnalysisPipeline") \
            .master("local[*]") \
            .config("spark.jars.packages", DELTA_PACKAGE) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.parquet.int96AsTimestamp", "true") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        print("Spark Session Initialized. Log level set to ERROR.")
        return spark
    except Exception as e:
        print(f"FATAL: Error initializing Spark Session: {e}")
        raise

# --- Main Execution Block (Unchanged) ---
# ... (Keep identical) ...
if __name__ == "__main__":
    INPUT_EXPLOITATION_PATH = "./data/exploitation/consolidated_municipal_annual"
    OUTPUT_ANALYSIS_DIR = "./data/analysis_outputs" # For plots/metrics
    OUTPUT_MODEL_DIR = "./models"                 # For model artifacts
    spark = None
    try:
        spark = get_spark_session()
        exploit_log_path = Path(INPUT_EXPLOITATION_PATH) / "_delta_log"
        if not exploit_log_path.is_dir():
            print(f"ERROR: Input Exploitation Delta table log not found at {exploit_log_path}")
            print("Please ensure the Exploitation Zone script ran successfully.")
        else:
            pipeline = DataAnalysisPipeline(spark=spark,
                                            input_path=INPUT_EXPLOITATION_PATH,
                                            output_dir=OUTPUT_ANALYSIS_DIR,
                                            model_dir=OUTPUT_MODEL_DIR)
            pipeline.run()
    except Exception as main_error:
        print(f"\nAn unexpected error occurred in the main execution block: {main_error}")
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping Spark Session.")
            spark.stop()
            print("Spark Session stopped.")