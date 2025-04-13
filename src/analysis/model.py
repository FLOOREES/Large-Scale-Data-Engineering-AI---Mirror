# data_analysis_pipeline.py

from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback
import datetime # Added import

# --- Core Imports ---
import pandas as pd
import numpy as np

# --- PySpark Imports ---
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
# Potentially add types if needed later for specific Spark operations

# --- ML Imports ---
from sklearn.model_selection import train_test_split # Or use time-based split #type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
# from sklearn.impute import SimpleImputer # REMOVED IMPUTER
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
import lightgbm as lgb # type: ignore

# --- Visualization (Optional Imports - Keep commented if not used immediately) ---
# import matplotlib.pyplot as plt
# import seaborn as sns

# --- Delta Lake Package Configuration (Keep consistent if loading directly) ---
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0"

class DataAnalysisPipeline:
    """
    A pipeline to perform data analysis and train a LightGBM model
    to predict average monthly rent using the consolidated exploitation data.
    BASELINE VERSION: Lets LightGBM handle NaNs after dropping rows missing target/lag-target.
    """

    def __init__(self, spark: SparkSession,
                 input_path: str = "./data/exploitation/consolidated_municipal_annual",
                 target_variable: str = "avg_monthly_rent_eur"):
        self.spark = spark
        self.input_path = input_path
        self.target_variable = target_variable
        self.id_cols = ["municipality_id", "municipality_name", "comarca_name"]
        self.time_col = "any"
        # Initial potential features (numeric - ALL included initially)
        self.numeric_features = [
            "total_annual_contracts", "salary_per_capita_eur", "population",
            "population_men", "population_women", "births_total",
            "population_spanish_nat", "surface_km2", "density_pop_km2",
            "longitude", "latitude", "unemployment_total_avg",
            "total_family_dwellings", "public_libraries_count",
            "sports_pavilions_count", "multi_sports_courts_count",
            "indoor_pools_count"
        ]
        self.categorical_features = ["comarca_name"] # Still keep as potential
        self.data_pd: pd.DataFrame = None
        self.features: List[str] = []
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.model: lgb.LGBMRegressor = None
        # self.imputer: SimpleImputer = None # REMOVED
        self.scaler: StandardScaler = None
        print("Data Analysis Pipeline Initialized.")
        print(f"  Input Delta Path: {self.input_path}")
        print(f"  Target Variable: {self.target_variable}")

    # --------------------------------------------------------------------------
    # 1. Data Loading (_load_data - unchanged)
    # --------------------------------------------------------------------------
    def _load_data(self):
        """Loads data from the Exploitation Zone Delta table into Pandas."""
        print("\n--- Step 1: Loading Data ---")
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
            mem_usage = self.data_pd.memory_usage(deep=True).sum() / (1024**2) # In MB
            print(f"Pandas DataFrame memory usage: {mem_usage:.2f} MB")
            if mem_usage > 1000:
                 print("WARNING: Pandas DataFrame size is large, consider sampling or Spark MLlib.")
        except Exception as e:
            print(f"!!! ERROR during data loading: {e}")
            traceback.print_exc()
            raise

    # --------------------------------------------------------------------------
    # 2. Exploratory Data Analysis (_analyze_data - unchanged)
    # --------------------------------------------------------------------------
    def _analyze_data(self):
        """Performs basic EDA on the loaded Pandas DataFrame."""
        if self.data_pd is None:
            print("Error: Data not loaded. Run _load_data() first.")
            return
        print("\n--- Step 2: Basic Data Analysis ---")
        print("\nDataFrame Info:")
        self.data_pd.info()
        print("\nDescriptive Statistics (Numeric Columns):")
        numeric_cols_pd = self.data_pd.select_dtypes(include=np.number).columns.tolist()
        print(self.data_pd[numeric_cols_pd].describe().to_string())
        print("\nMissing Value Counts:")
        missing_counts = self.data_pd.isnull().sum()
        print(missing_counts[missing_counts > 0].to_string())
        # print("\nUnique values in 'comarca_name':") # Keep example
        # print(self.data_pd['comarca_name'].nunique())
        # print(self.data_pd['comarca_name'].value_counts().head())
        print("\nBasic Data Analysis complete.")

    # --------------------------------------------------------------------------
    # 3. Feature Engineering (_feature_engineering - MODIFIED dropna logic)
    # --------------------------------------------------------------------------
    def _feature_engineering(self):
        """Creates new features, particularly time-based features (lags)."""
        if self.data_pd is None:
            print("Error: Data not loaded. Run _load_data() first.")
            return
        print("\n--- Step 3: Feature Engineering ---")

        print(f"Creating lag features for target variable '{self.target_variable}'...")
        self.data_pd = self.data_pd.sort_values(by=[*self.id_cols, self.time_col]).reset_index(drop=True)
        lag_target_col = f"{self.target_variable}_lag1"
        self.data_pd[lag_target_col] = self.data_pd.groupby(self.id_cols)[self.target_variable].shift(1)

        # --- Define final feature list ---
        # Include ALL original numeric + engineered lags initially
        self.features = self.numeric_features + [lag_target_col]
        self.features = [f for f in self.features if f in self.data_pd.columns]
        if self.time_col not in self.features:
             self.features.append(self.time_col)
        print(f"Engineered features created. Initial candidate features: {self.features}")

        # --- Handle NaNs introduced ONLY by essential lags/target ---
        # Define columns that MUST NOT be NaN for a row to be useful for training/evaluation
        essential_cols_for_dropna = [self.target_variable, lag_target_col] # Add others if strictly necessary
        print(f"Dropping rows only if NaNs exist in essential columns: {essential_cols_for_dropna}")
        initial_rows = len(self.data_pd)
        self.data_pd = self.data_pd.dropna(subset=essential_cols_for_dropna).reset_index(drop=True)
        rows_after_drop = len(self.data_pd)
        print(f"Dropped {initial_rows - rows_after_drop} rows based on NaNs in essential columns.")
        print(f"DataFrame shape after essential drop: {self.data_pd.shape}")

        # Check remaining NaNs in features (informational)
        print("\nMissing values REMAINING in candidate features (to be handled by LightGBM):")
        remaining_missing = self.data_pd[self.features].isnull().sum()
        print(remaining_missing[remaining_missing > 0].to_string())

        print("Feature Engineering complete.")

    # --------------------------------------------------------------------------
    # 4. Data Preprocessing (_preprocess_data - REMOVED Imputer, FIXED warnings)
    # --------------------------------------------------------------------------
    def _preprocess_data(self):
        """Splits data, handles categoricals (optional), and scales features."""
        if self.data_pd is None or not self.features:
             print("Error: Data not loaded or features not defined. Run previous steps.")
             return
        print("\n--- Step 4: Preprocessing Data ---")

        # --- 4a. Split Data (Time-Based) ---
        print("Splitting data into training and testing sets (time-based)...")
        test_year = self.data_pd[self.time_col].max()
        train_df = self.data_pd[self.data_pd[self.time_col] < test_year].copy() # Use .copy()
        test_df = self.data_pd[self.data_pd[self.time_col] == test_year].copy() # Use .copy()

        if len(train_df) == 0 or len(test_df) == 0:
             raise ValueError(f"Train or test set is empty after splitting on year {test_year}.")

        self.X_train = train_df[self.features]
        self.y_train = train_df[self.target_variable]
        self.X_test = test_df[self.features]
        self.y_test = test_df[self.target_variable]
        print(f"Data split: Train shape={self.X_train.shape}, Test shape={self.X_test.shape}")

        # --- 4b. Handle Missing Values (REMOVED IMPUTATION) ---
        print("Skipping explicit missing value imputation (LightGBM will handle NaNs).")

        # --- 4c. Encode Categorical Features (Placeholder - Unchanged) ---
        # ... (keep placeholder as before) ...
        print("Skipping categorical encoding for baseline.")


        # --- 4d. Scale Numeric Features (Post-Split) ---
        print("Scaling numeric features...")
        # Identify numeric features AFTER splitting and potential encoding
        final_numeric_features = self.X_train.select_dtypes(include=np.number).columns.tolist()
        if not final_numeric_features:
             print("No numeric features found to scale.")
        else:
            self.scaler = StandardScaler()
            print(f"Fitting scaler on training data (features: {final_numeric_features})...")
            self.scaler.fit(self.X_train[final_numeric_features]) # Fit only on train

            print("Transforming training and testing data using .loc...")
            # Use .loc to avoid SettingWithCopyWarning
            self.X_train.loc[:, final_numeric_features] = self.scaler.transform(self.X_train[final_numeric_features])
            self.X_test.loc[:, final_numeric_features] = self.scaler.transform(self.X_test[final_numeric_features])
            print("Feature scaling complete.")

        print("Preprocessing complete.")


    # --------------------------------------------------------------------------
    # 5. Model Training (_train_model - unchanged)
    # --------------------------------------------------------------------------
    def _train_model(self):
        """Trains the LightGBM model."""
        if self.X_train is None or self.y_train is None:
             print("Error: Training data not prepared. Run preprocessing first.")
             return
        print("\n--- Step 5: Training LightGBM Model ---")
        params = {
            'objective': 'regression_l1', 'metric': ['mae', 'rmse'],
            'n_estimators': 1000, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
            'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42
        }
        print(f"Training LGBMRegressor with parameters: {params}")
        self.model = lgb.LGBMRegressor(**params)
        start_time = datetime.datetime.now()
        # LightGBM handles NaNs automatically if imputer was removed
        self.model.fit(self.X_train, self.y_train)
        end_time = datetime.datetime.now()
        print(f"Model training complete in {end_time - start_time}.")


    # --------------------------------------------------------------------------
    # 6. Prediction and Evaluation (_evaluate_model - FIXED RMSE calculation)
    # --------------------------------------------------------------------------
    def _evaluate_model(self):
        """Makes predictions and evaluates the model on the test set."""
        if self.model is None or self.X_test is None or self.y_test is None:
            print("Error: Model not trained or test data not ready.")
            return
        print("\n--- Step 6: Evaluating Model ---")

        print("Making predictions on the test set...")
        y_pred = self.model.predict(self.X_test)

        # Calculate metrics - CORRECTED RMSE
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred) # Calculate MSE first
        rmse = np.sqrt(mse)                          # Calculate RMSE from MSE
        r2 = r2_score(self.y_test, y_pred)

        print("\nEvaluation Metrics on Test Set:")
        print(f"  Mean Absolute Error (MAE):  {mae:.2f}")
        print(f"  Mean Squared Error (MSE):   {mse:.2f}") # Also report MSE
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"  R-squared (R²):           {r2:.4f}")

        # Feature Importance (keep as before)
        try:
            print("\nFeature Importances (Top 15):")
            feature_imp = pd.DataFrame({'Value': self.model.feature_importances_, 'Feature': self.X_train.columns})
            feature_imp = feature_imp.sort_values(by="Value", ascending=False).head(15)
            print(feature_imp.to_string(index=False))
        except Exception as fi_e:
            print(f"Could not display feature importances: {fi_e}")


    # --------------------------------------------------------------------------
    # Pipeline Orchestration (run - unchanged)
    # --------------------------------------------------------------------------
    def run(self):
        """Runs the full data analysis and modeling pipeline."""
        print("==================================================")
        print("=== Starting Data Analysis and Prediction Pipeline ===")
        print("==================================================")
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

# --- Spark Session Creation Helper (Same as before) ---
# ... (keep identical)
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

# --- Main Execution Block (Same as before) ---
# ... (keep identical)
if __name__ == "__main__":
    INPUT_EXPLOITATION_PATH = "./data/exploitation/consolidated_municipal_annual"
    spark = None
    try:
        spark = get_spark_session()
        exploit_log_path = Path(INPUT_EXPLOITATION_PATH) / "_delta_log"
        if not exploit_log_path.is_dir():
            print(f"ERROR: Input Exploitation Delta table log not found at {exploit_log_path}")
            print("Please ensure the Exploitation Zone script ran successfully.")
        else:
            pipeline = DataAnalysisPipeline(spark=spark, input_path=INPUT_EXPLOITATION_PATH)
            pipeline.run()
    except Exception as main_error:
        print(f"\nAn unexpected error occurred in the main execution block: {main_error}")
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping Spark Session.")
            spark.stop()
            print("Spark Session stopped.")