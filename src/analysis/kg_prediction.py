# src/analysis/kg_link_prediction.py

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
import traceback
import joblib
from typing import List, Dict, Any

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# PyKEEN imports for loading embeddings
from pykeen.triples import TriplesFactory

# PySpark for loading initial data
from pyspark.sql import SparkSession

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define key paths
EXPLOITATION_DATA_PATH = Path("./data/exploitation/municipal_annual")
EMBEDDINGS_DIR = Path("./data/analysis/model_embeddings/comparison_run")
ANALYSIS_OUTPUT_DIR = Path("./data/analysis/rent_prediction_model")
MODELS_DIR = Path("./models")

class RentPredictionPipeline:
    """
    A pipeline to train a LightGBM model to predict rent by combining
    time-series features with pre-trained Knowledge Graph embeddings.
    """

    def __init__(self, spark: SparkSession, best_experiment_id: str,
                 target_variable: str = "avg_monthly_rent"):
        self.spark = spark
        self.best_experiment_id = best_experiment_id
        self.target_variable = target_variable
        
        self.exploitation_path = EXPLOITATION_DATA_PATH
        self.embeddings_path = EMBEDDINGS_DIR / self.best_experiment_id
        self.triples_path = EMBEDDINGS_DIR / "triples.tsv"
        self.output_dir = ANALYSIS_OUTPUT_DIR
        self.models_dir = MODELS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.id_cols = ["municipality_id"]
        self.time_col = "any"
        self.categorical_features = ["comarca_name"]
        
        logger.info("Rent Prediction Pipeline Initialized.")
        logger.info(f"  Using embeddings from: {self.best_experiment_id}")
        logger.info(f"  Target variable: {self.target_variable}")

    def _load_and_merge_data(self) -> pd.DataFrame:
        logger.info(f"Loading main data from Delta table: {self.exploitation_path}")
        main_df = self.spark.read.format("delta").load(str(self.exploitation_path)).toPandas()
        logger.info(f"Loaded {len(main_df)} rows of main data.")

        logger.info(f"Loading embeddings for experiment: {self.best_experiment_id}")
        
        model_path = self.embeddings_path / 'trained_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if not self.triples_path.exists():
            raise FileNotFoundError(f"Shared triples file not found at {self.triples_path}")
        triples_factory = TriplesFactory.from_path(self.triples_path)
        
        entity_embeddings_layer = model.entity_representations[0]
        
        if hasattr(entity_embeddings_layer, '_embeddings'):
            entity_embeddings_tensor = entity_embeddings_layer._embeddings.weight.data
        else:
            entity_embeddings_tensor = entity_embeddings_layer.weight.data

        embedding_dim = entity_embeddings_tensor.shape[1]
        id_to_entity = {v: k for k, v in triples_factory.entity_to_id.items()}
        
        entity_data = []
        for i in range(len(id_to_entity)):
            uri = id_to_entity.get(i)
            if uri and '/municipality/' in uri:
                entity_id = uri.split('/municipality/')[-1]
                embedding_vector = entity_embeddings_tensor[i].numpy()
                entity_data.append([entity_id] + list(embedding_vector))
        
        emb_columns = ['municipality_id'] + [f'emb_dim_{j}' for j in range(embedding_dim)]
        embeddings_df = pd.DataFrame(entity_data, columns=emb_columns)
        logger.info(f"Loaded and processed {len(embeddings_df)} entity embeddings.")
        
        logger.info("Merging main data with embeddings...")
        merged_df = pd.merge(main_df, embeddings_df, on="municipality_id", how="inner")
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        
        return merged_df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("--- Step 2: Feature Engineering ---")
        df = df.sort_values(by=[*self.id_cols, self.time_col]).reset_index(drop=True)
        
        cols_for_lags = [self.target_variable, 'total_contracts', 'income_per_capita']
        self.engineered_features = []

        for col in cols_for_lags:
            if col in df.columns:
                for lag in [1, 2]:
                    lag_col_name = f"{col}_lag{lag}"
                    df[lag_col_name] = df.groupby(self.id_cols)[col].shift(lag)
                    self.engineered_features.append(lag_col_name)

        lag1_target = f"{self.target_variable}_lag1"
        if lag1_target in df.columns:
            df[f"{self.target_variable}_roll_mean3_lag1"] = df.groupby(self.id_cols)[lag1_target].rolling(window=3, min_periods=1).mean().reset_index(level=self.id_cols, drop=True)
            self.engineered_features.append(f"{self.target_variable}_roll_mean3_lag1")
        
        logger.info("Time-based features created.")
        return df

    def _preprocess_and_split(self, df: pd.DataFrame):
        """
        Defines the final feature set, handles NaNs, splits data, and scales features.
        """
        logger.info("--- Step 3: Preprocessing and Splitting Data ---")

        # --- Handle Missing Data FIRST ---
        essential_cols = [self.target_variable, f"{self.target_variable}_lag1"]
        initial_rows = len(df)
        df.dropna(subset=essential_cols, inplace=True)
        logger.info(f"Dropped {initial_rows - len(df)} rows due to missing target or lag1.")

        # --- Define Final Feature Set ---
        all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_features = [col for col in all_numeric_cols if col != self.target_variable]
        self.features = sorted(list(set(numeric_features + self.categorical_features)))
        self.features = [f for f in self.features if f in df.columns]
        logger.info(f"Total features for model: {len(self.features)}")
        
        # --- Time-based Split ---
        if df[self.time_col].nunique() < 3:
            raise ValueError("Not enough unique years for train-val-test split.")

        test_year = df[self.time_col].max()
        train_val_df = df[df[self.time_col] < test_year].copy()
        test_df = df[df[self.time_col] == test_year].copy()
        
        val_year = train_val_df[self.time_col].max()
        val_df = train_val_df[train_val_df[self.time_col] == val_year].copy()
        train_df = train_val_df[train_val_df[self.time_col] < val_year].copy()
        
        self.X_train, self.y_train = train_df[self.features].copy(), train_df[self.target_variable].copy()
        self.X_val, self.y_val = val_df[self.features].copy(), val_df[self.target_variable].copy()
        self.X_test, self.y_test = test_df[self.features].copy(), test_df[self.target_variable].copy()

        logger.info(f"Data split complete: Train={self.X_train.shape}, Val={self.X_val.shape}, Test={self.X_test.shape}")

        if self.X_train.empty or self.X_val.empty or self.X_test.empty:
            raise ValueError("One of the data splits is empty.")

        # --- Handle Categorical Features ---
        for col in self.categorical_features:
            if col in self.features:
                self.X_train[col] = self.X_train[col].astype('category')
                self.X_val[col] = pd.Categorical(self.X_val[col], categories=self.X_train[col].cat.categories)
                self.X_test[col] = pd.Categorical(self.X_test[col], categories=self.X_train[col].cat.categories)
        
        # --- Scale Numeric Features ---
        numeric_cols_to_scale = self.X_train.select_dtypes(include=np.number).columns.tolist()
        self.scaler = StandardScaler()
        
        self.X_train[numeric_cols_to_scale] = self.scaler.fit_transform(self.X_train[numeric_cols_to_scale])
        self.X_val[numeric_cols_to_scale] = self.scaler.transform(self.X_val[numeric_cols_to_scale])
        self.X_test[numeric_cols_to_scale] = self.scaler.transform(self.X_test[numeric_cols_to_scale])
        
        joblib.dump(self.scaler, self.models_dir / 'rent_prediction_scaler.joblib')
        logger.info("Numeric features scaled and scaler saved.")

    def _train_model(self):
        """
        Trains the LightGBM model. First, it uses a validation set for early
        stopping to find the optimal number of boosting rounds. Then, it
        retrains a new model on the full training + validation data for that
        optimal number of rounds.
        """
        logger.info("--- Step 4: Training LightGBM Model ---")
        
        # --- Define Parameters ---
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000,
            'learning_rate': 0.02, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 1, 'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42
        }

        # --- Part A: Find the optimal number of iterations ---
        logger.info("Step 4a: Finding best iteration using validation set...")
        
        temp_model = lgb.LGBMRegressor(**params)
        
        temp_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(100, verbose=True)]
        )
        
        best_iteration = temp_model.best_iteration_
        if best_iteration is None:
            logger.warning("Early stopping did not trigger. Using default n_estimators.")
            best_iteration = params['n_estimators']
        
        logger.info(f"Optimal number of iterations found: {best_iteration}")

        # --- Part B: Re-train a new model on the combined data ---
        logger.info("Step 4b: Retraining final model on combined train+validation data...")
        
        # Combine the original training and validation sets
        X_train_full = pd.concat([self.X_train, self.X_val], ignore_index=True)
        y_train_full = pd.concat([self.y_train, self.y_val], ignore_index=True)

        # Update params with the optimal number of estimators
        final_params = params.copy()
        final_params['n_estimators'] = best_iteration

        self.model = lgb.LGBMRegressor(**final_params)
        
        # Handle categoricals for the full training data
        for col in self.categorical_features:
            if col in X_train_full.columns:
                X_train_full[col] = X_train_full[col].astype('category')

        self.model.fit(X_train_full, y_train_full,
                       categorical_feature=[col for col in self.categorical_features if col in X_train_full.columns]
                      )
        
        logger.info("Final model has been retrained on all available data (excluding test set).")
        
        # Save the final retrained model
        joblib.dump(self.model, self.models_dir / 'rent_predictor_lgbm.joblib')
        logger.info(f"Model saved to: {self.models_dir / 'rent_predictor_lgbm.joblib'}")

    def _evaluate_model(self):
        logger.info("--- Step 5: Evaluating Model on Test Set ---")
        y_pred = self.model.predict(self.X_test)

        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

        logger.info(f"  R-squared (R²): {r2:.4f}")
        logger.info(f"  Mean Absolute Error (MAE): {mae:.2f} €")
        logger.info(f"  Root Mean Squared Error (RMSE): {rmse:.2f} €")

        lgb.plot_importance(self.model, max_num_features=30, figsize=(10, 12), importance_type='gain')
        plt.title(f'Top 30 Feature Importance for Rent Prediction (Test R²={r2:.3f})')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.show(block=False)

        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=self.y_test, y=y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.title('Prediction vs. Actual Rent')
        plt.xlabel('Actual Rent (€)')
        plt.ylabel('Predicted Rent (€)')
        plt.savefig(self.output_dir / 'prediction_vs_actual.png')
        plt.show(block=False)
        
        metrics_dict = {
            'r2_score': r2,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse
        }
        
        metrics_path = self.output_dir / 'prediction_results.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Prediction metrics saved to: {metrics_path}")

        plt.pause(1)

    def run(self):
        logger.info("="*50)
        logger.info("=== Starting Rent Prediction Pipeline (w/ KG Embeddings) ===")
        logger.info("="*50)
        try:
            full_data = self._load_and_merge_data()
            featured_data = self._feature_engineering(full_data)
            self._preprocess_and_split(featured_data)
            self._train_model()
            self._evaluate_model()
            logger.info("=== Pipeline Execution Successfully Completed ===")
        except Exception as e:
            logger.error(f"!!! Pipeline Execution Failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    from spark_session import get_spark_session
    spark = None
    try:
        spark = get_spark_session()
        
        best_kge_model_id = 'TransH_dim_128' 
        pipeline = RentPredictionPipeline(spark=spark, best_experiment_id=best_kge_model_id)
        pipeline.run()

    except Exception as main_error:
        print(f"\nAn unexpected error occurred in the main execution block: {main_error}")
        traceback.print_exc()
    finally:
        if spark:
            print("\nStopping Spark Session.")
            spark.stop()
            print("Spark Session stopped.")