# src/analysis/kg_embeddings.py

import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from typing import List, Dict, Any

# PyKEEN imports
from pykeen.pipeline import pipeline, PipelineResult
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import pykeen.version

# RDFLib for initial graph processing
import rdflib
from rdflib import Graph

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define key paths
GRAPH_PATH = Path("./data/exploitation/knowledge_graph.ttl")
OUTPUT_DIR = Path("./data/analysis/model_embeddings")


class KGEmbeddings:
    """
    A robust pipeline for training, loading, and comparing multiple Knowledge Graph
    embedding models with different hyperparameters using PyKEEN.
    """

    def __init__(self, model_configs: List[Dict[str, Any]], graph_path: Path = GRAPH_PATH, epochs: int = 100, batch_size: int = 256, force_training: bool = False):
        if not isinstance(model_configs, list) or not model_configs:
            raise ValueError("model_configs must be a non-empty list of dictionaries.")

        self.graph_path = graph_path
        self.model_configs = model_configs
        self.epochs = epochs
        self.batch_size = batch_size
        self.force_training = force_training
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.run_output_dir = OUTPUT_DIR / "comparison_run"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.triples_path = self.run_output_dir / "triples.tsv"
        
        logger.info(f"Pipeline initialized for {len(model_configs)} experiments on device '{self.device.upper()}'")
        logger.info(f"Force training set to: {self.force_training}")

    def _prepare_data(self, create_inverses: bool = True):
        """
        Loads, filters, and splits the data. This can be done with or without inverse triples.
        """
        logger.info(f"--- Data Preparation (create_inverse_triples={create_inverses}) ---")
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Knowledge Graph file not found at {self.graph_path}.")

        # Check if the raw triples file already exists to save time
        if not self.triples_path.exists() or self.force_training:
            g = Graph().parse(self.graph_path, format="turtle")
            entity_triples = [
                (str(s), str(p), str(o))
                for s, p, o in g
                if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef)
            ]
            with open(self.triples_path, "w", encoding='utf-8') as f:
                for s, p, o in entity_triples:
                    f.write(f"{s}\t{p}\t{o}\n")
            logger.info(f"Created raw triples file at {self.triples_path}")

        tf = TriplesFactory.from_path(self.triples_path, create_inverse_triples=create_inverses)
        return tf.split([0.8, 0.1, 0.1], random_state=42)

    def run(self) -> Dict[str, Any]:
        all_results = {}

        for config in self.model_configs:
            model_name = config['name']
            embedding_dim = config['dim']
            experiment_id = f"{model_name}_dim_{embedding_dim}"
            model_output_dir = self.run_output_dir / experiment_id
            
            # **FIXED**: Dynamically decide whether to create inverse triples based on model
            create_inverses = model_name != 'RGCN'
            training_set, validation_set, testing_set = self._prepare_data(create_inverses=create_inverses)
            
            if not self.force_training and (model_output_dir / 'trained_model.pkl').exists():
                logger.info(f"Found pre-trained model for '{experiment_id}'. Loading from: {model_output_dir}")
                try:
                    model = torch.load(model_output_dir / 'trained_model.pkl', map_location=self.device, weights_only=False)
                    evaluator = RankBasedEvaluator()
                    metric_results = evaluator.evaluate(
                        model=model, mapped_triples=testing_set.mapped_triples,
                        additional_filter_triples=[training_set.mapped_triples, validation_set.mapped_triples],
                        batch_size=self.batch_size, device=self.device
                    )
                    all_results[experiment_id] = {'metric_results': metric_results}
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load model '{experiment_id}'. Will retrain. Error: {e}")

            logger.info(f"\n{'='*25} Training Experiment: {experiment_id} {'='*25}")
            try:
                result = pipeline(
                    training=training_set, validation=validation_set, testing=testing_set,
                    model=model_name,
                    model_kwargs=dict(embedding_dim=embedding_dim),
                    evaluation_kwargs=dict(batch_size=self.batch_size),
                    training_kwargs=dict(num_epochs=self.epochs, batch_size=self.batch_size, use_tqdm_batch=False),
                    stopper='early', stopper_kwargs=dict(frequency=5, metric='inverse_harmonic_mean_rank', patience=3),
                    random_seed=42, device=self.device
                )
                result.save_to_directory(model_output_dir)
                all_results[experiment_id] = result
            except Exception as e:
                logger.error(f"Failed to run pipeline for '{experiment_id}'.", exc_info=True)
        
        return all_results
    
    def _plot_heatmap(self, pivot_df: pd.DataFrame):
        plt.figure(figsize=(12, len(pivot_df) * 0.8 + 2))
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="Blues", linewidths=.5)
        plt.title('KGE Model Performance Heatmap', fontsize=16)
        plt.xlabel('Evaluation Metric')
        plt.ylabel('Experiment')
        plt.xticks(rotation=20, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.run_output_dir / 'model_comparison_heatmap.png')
        logger.info(f"Comparison heatmap saved to: {self.run_output_dir / 'model_comparison_heatmap.png'}")
        plt.show(block=False)

    def _plot_faceted_barplot(self, results_df: pd.DataFrame, metrics_to_plot: List[str]):
        """Generates a faceted bar plot with improved legend placement and bar appearance."""
        plot_data = results_df[results_df['Metric'].isin(metrics_to_plot)]
        
        g = sns.catplot(
            data=plot_data,
            x='Model', y='Value', hue='Dimension',
            col='Metric', kind='bar',
            col_wrap=3, sharey=False,
            col_order=metrics_to_plot,
            height=5,
            palette="tab10",
        )

        g.figure.suptitle('Overall Model Performance Comparison', y=1.03, fontsize=20)
        g.set_axis_labels("Model", "Score")
        g.set_titles("{col_name}")

        for ax in g.axes.flat:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f', fontsize=9, rotation=90, padding=5)
            ax.margins(y=0.2)
            # Ensure x-axis labels are visible on all subplots
            ax.tick_params(axis='x', which='both', labelbottom=True)
        
        # Move the legend outside the plot area for clarity
        sns.move_legend(g, "upper right", bbox_to_anchor=(1.0, 0.95))

        # Adjust layout to prevent overlap and add spacing
        g.figure.subplots_adjust(hspace=0.8, wspace=0.3) # Increase vertical spacing more to prevent label overlap
        plt.tight_layout(rect=(0, 0, 1, 0.96)) # Adjust overall layout, leaving space for suptitle

        plot_path = self.run_output_dir / 'model_comparison_faceted_barplot.png'
        # Save with bbox_inches='tight' to ensure the external legend is included
        plt.savefig(plot_path, bbox_inches='tight')
        logger.info(f"Faceted comparison bar plot saved to: {plot_path}")
        plt.show(block=False)

    def compare_and_plot_results(self, results: Dict[str, Any]):
        if not results:
            logger.warning("No model results to compare.")
            return

        logger.info("\n--- Comparing Model Performance ---")
        all_metrics_data = []
        for experiment_id, result in results.items():
            try:
                if isinstance(result, PipelineResult):
                    metrics_df = result.metric_results.to_df()
                elif isinstance(result, dict) and 'metric_results' in result:
                    metrics_df = result['metric_results'].to_df()
                else: continue
                
                base_filter = (metrics_df['Side'] == 'both') & (metrics_df['Rank_type'] == 'realistic')
                if 'Dataset' in metrics_df.columns:
                    base_filter &= (metrics_df['Dataset'] == 'testing')
                
                test_metrics = metrics_df[base_filter].copy()
                if test_metrics.empty: continue

                model_name, dim_str = experiment_id.split('_dim_')
                test_metrics['Model'] = model_name
                test_metrics['Dimension'] = int(dim_str)
                test_metrics['Experiment'] = f"{model_name} (d={dim_str})"
                all_metrics_data.append(test_metrics)
                logger.info(f"Successfully processed metrics for {experiment_id}.")

            except Exception as e:
                logger.error(f"Error processing metrics for {experiment_id}.", exc_info=True)

        if not all_metrics_data:
            logger.error("Failed to extract metrics from any model. Aborting comparison.")
            return

        results_df = pd.concat(all_metrics_data, ignore_index=True)
        pivot_df = results_df.pivot_table(index='Experiment', columns='Metric', values='Value', aggfunc='first')
        
        mrr_metric_name = 'inverse_harmonic_mean_rank'
        final_metrics_order = [mrr_metric_name, 'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10']
        final_metrics_existing = [m for m in final_metrics_order if m in pivot_df.columns]
        
        if mrr_metric_name in pivot_df.columns:
            pivot_df = pivot_df.sort_values(by=mrr_metric_name, ascending=False)
        
        metric_display_names = {
            'inverse_harmonic_mean_rank': 'IHMR',
            'hits_at_1': 'Hits@1',
            'hits_at_3': 'Hits@3',
            'hits_at_5': 'Hits@5',
            'hits_at_10': 'Hits@10',
        }
        
        display_pivot_df = pivot_df.rename(columns=metric_display_names)
        display_metrics_order = [metric_display_names.get(m, m) for m in final_metrics_existing]
        
        print("\n\n" + "="*20 + " Final Model Comparison " + "="*20)
        print(display_pivot_df[display_metrics_order].to_string(float_format="%.4f"))

        display_results_df = results_df.copy()
        display_results_df['Metric'] = display_results_df['Metric'].map(metric_display_names).fillna(display_results_df['Metric'])

        self._plot_heatmap(display_pivot_df[display_metrics_order])
        self._plot_faceted_barplot(display_results_df, display_metrics_order)
        
        plt.pause(1)


if __name__ == '__main__':
    print(f"Using PyKEEN version: {pykeen.version.get_version()}")
    print(f"Using PyTorch version: {torch.__version__}")

    # TransE, TransH, TransR, DistMult, RotatE, ComplEx, RGCN, CompGCN in 64, 128, 256 dimensions
    experiments_to_run = [
        {'name': 'TransE', 'dim': 64},
        {'name': 'TransH', 'dim': 64},
        {'name': 'TransR', 'dim': 64},
        {'name': 'DistMult', 'dim': 64},
        {'name': 'RotatE', 'dim': 64},
        {'name': 'ComplEx', 'dim': 64},
        {'name': 'RGCN', 'dim': 64},
        {'name': 'CompGCN', 'dim': 64},
        {'name': 'TransE', 'dim': 128},
        {'name': 'TransH', 'dim': 128},
        {'name': 'TransR', 'dim': 128},
        {'name': 'DistMult', 'dim': 128},
        {'name': 'RotatE', 'dim': 128},
        {'name': 'ComplEx', 'dim': 128},
        {'name': 'RGCN', 'dim': 128},
        {'name': 'CompGCN', 'dim': 128},
        {'name': 'TransE', 'dim': 256},
        {'name': 'TransH', 'dim': 256},
        {'name': 'TransR', 'dim': 256},
        {'name': 'DistMult', 'dim': 256},
        {'name': 'RotatE', 'dim': 256},
        {'name': 'ComplEx', 'dim': 256},
        {'name': 'RGCN', 'dim': 256},
        {'name': 'CompGCN', 'dim': 256}
    ]

    print("\n--- Running Experiment: Load if available, train if missing ---")
    try:
        pipeline_runner = KGEmbeddings(
            model_configs=experiments_to_run,
            epochs=300,
            batch_size=512,
            force_training=False
        )
        
        results = pipeline_runner.run()
        pipeline_runner.compare_and_plot_results(results)

    except Exception as e:
        logger.error("An error occurred during the main execution block.", exc_info=True)