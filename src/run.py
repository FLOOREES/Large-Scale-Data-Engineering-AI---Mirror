# src/run.py

"""Main file to run the entire pipeline of the project."""

from setup import make_folder_structure
from pipeline import Pipeline

if __name__ == "__main__":
    make_folder_structure()

    # --- Configuration for the Embeddings Training/Evaluation Stage ---
    # Define which KGE models and dimensions to experiment with.
    kg_embeddings_config = {
        'model_configs': [
            {'name': 'TransH', 'dim': 128},
        ],
        'epochs': 300,
        'batch_size': 512,
        'force_training': False,
    }

    # --- Configuration for the Final Rent Prediction Stage ---
    # Specify which of the trained embeddings should be used as features.
    kg_prediction_config = {
        'best_experiment_id': 'TransH_dim_128', # Assumes this was the best from the experiments
    }

    # --- Pipeline Execution ---
    # Control which parts of the project to run.
    # start_stage: 1=Landing, 2=Formatted, 3=Trusted, 4=Exploitation, 5=Analysis
    # analysis_parts: A list containing any of "embeddings", "query", "prediction"
    
    pipeline = Pipeline(
        start_stage=5,
        max_stage=5,
        analysis_parts=["prediction"], # Run only the final prediction model
        kg_embeddings_config=kg_embeddings_config,
        kg_prediction_config=kg_prediction_config
    )
    
    pipeline.run()
