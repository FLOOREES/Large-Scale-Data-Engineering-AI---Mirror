"""Main file to run the entire pipeline of the project."""

from setup import make_folder_structure
from pipeline import Pipeline

if __name__ == "__main__":
	make_folder_structure()
	# Stages:
	# 1. Landing Zone
	# 2. Formatted Zone
	# 3. Trusted Zone
	# 4. Exploitation Zone
	# 5. Analysis (model, visualizer, or both)


	experiments_to_run = [
        {'name': 'TransE', 'dim': 64},
    ]

	kg_embeddings_config = {
		'model_configs': experiments_to_run,
		'epochs': 1,
		'batch_size': 512,
		'force_training': False,
		'create_plots_in_run': True,
	}

	pipeline = Pipeline(start_stage=5, max_stage=5, analysis="embeddings", kg_embeddings_config=kg_embeddings_config)
	pipeline.run()
