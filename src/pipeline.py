# src/pipeline.py

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from spark_session import get_spark_session
from landing.landing import LandingZone
from formatted.formatted import FormattedZone
from trusted.trusted import TrustedZone
from exploitation.exploitation import ExploitationZone
from analysis.kg_embeddings import KGEmbeddings
from analysis.kg_query import KGQueryPipeline
from analysis.kg_graphrag import GraphRAGPipeline
from analysis.kg_prediction import RentPredictionPipeline

class Pipeline:
    """
    Main class to orchestrate the entire data and analysis pipeline.
    Initializes components just-in-time to ensure dependencies from
    previous stages are met.
    """
    def __init__(self, start_stage: int = 1, max_stage: int = 5,
                 analysis_parts: List[str] = None,
                 kg_query_config: Dict[str, Any] = None,
                 kg_rag_config: Dict[str, Any] = None,
                 kg_embeddings_config: Dict[str, Any] = None,
                 kg_prediction_config: Dict[str, Any] = None):
        
        # --- Validate Inputs ---
        self.analysis_parts = analysis_parts or []
        allowed_parts = {"query", "rag_query", "embeddings", "prediction"}
        if not all(part in allowed_parts for part in self.analysis_parts):
            raise ValueError(f"Invalid analysis_parts. Allowed values are: {allowed_parts}")
        
        # --- Store Configs and Parameters ---
        self.kg_query_config = kg_query_config
        self.kg_rag_config = kg_rag_config
        self.kg_embeddings_config = kg_embeddings_config
        self.kg_prediction_config = kg_prediction_config
        self.start_stage = start_stage
        self.max_stage = max_stage
        
        # --- Spark Session Management ---
        # Initialize Spark only if needed for stages that use it.
        self.spark = None
        if (self.start_stage <= 4 and self.max_stage >= 2) or ("prediction" in self.analysis_parts):
            self.spark = get_spark_session()

    def run(self):
        """
        Executes the configured pipeline stages by instantiating them on-the-fly.
        """
        try:
            if self.start_stage <= 1 <= self.max_stage:
                print("\n" + "="*100 + "\nPIPELINE: STARTING LANDING ZONE (STAGE 1)\n" + "="*100)
                landing = LandingZone()
                landing.run()
            
            if self.start_stage <= 2 <= self.max_stage:
                print("\n" + "="*100 + "\nPIPELINE: STARTING FORMATTED ZONE (STAGE 2)\n" + "="*100)
                formatted = FormattedZone(spark=self.spark)
                formatted.run()
            
            if self.start_stage <= 3 <= self.max_stage:
                print("\n" + "="*100 + "\nPIPELINE: STARTING TRUSTED ZONE (STAGE 3)\n" + "="*100)
                trusted = TrustedZone(spark=self.spark)
                trusted.run()
            
            if self.start_stage <= 4 <= self.max_stage:
                print("\n" + "="*100 + "\nPIPELINE: STARTING EXPLOITATION ZONE (STAGE 4)\n" + "="*100)
                exploitation = ExploitationZone(spark=self.spark)
                exploitation.run()

            if self.start_stage <= 5 <= self.max_stage:
                self._run_analysis()

        finally:
            print("\n" + "="*100 + "\nPIPELINE ENDED\n" + "="*100)
            if self.spark is not None:
                self.spark.stop()
                print("Spark Session stopped.")

    def _run_analysis(self):
        """Helper method to run the various parts of the analysis stage."""
        print("\n" + "="*100 + f"\nPIPELINE: STARTING ANALYSIS (STAGE 5) - Parts: {self.analysis_parts}\n" + "="*100)
        
        analysis_execution_order = ["query", "rag_query", "embeddings", "prediction"]

        for part in analysis_execution_order:
            if part not in self.analysis_parts:
                continue
            
            if part == "query":
                print("\n" + "-"*40 + " Running SPARQL Query Analysis " + "-"*40)
                if not self.kg_query_config or not self.kg_query_config.get('queries'):
                    print("Skipping: No queries found in kg_query_config.")
                    continue
                
                analysis_query = KGQueryPipeline(graph_path=Path("./data/exploitation/knowledge_graph.ttl"))
                for query_info in self.kg_query_config['queries']:
                    print(f"\n--- Executing Query: {query_info.get('name', 'Unnamed Query')} ---")
                    results_df = analysis_query.run(sparql_query=query_info['sparql'])
                    print("\n--- Results ---")
                    if results_df.empty:
                        print("No results found for this query.")
                    else:
                        pd.set_option('display.width', 120); pd.set_option('display.max_rows', 100)
                        print(results_df)

            elif part == "rag_query":
                print("\n" + "-"*40 + " Running GraphRAG Analysis " + "-"*40)
                if not self.kg_rag_config or not self.kg_rag_config.get('questions'):
                    print("Skipping: No questions found in kg_rag_config.")
                    continue
                
                analysis_rag = GraphRAGPipeline(
                    turtle_path=Path("./data/exploitation/knowledge_graph.ttl"),
                    llm_config=self.kg_rag_config.get('llm_config', {})
                )
                for q in self.kg_rag_config['questions']:
                    answer = analysis_rag.run(q)
                    print(f"\n> Question: {q}")
                    print("\n< Answer:")
                    print(answer)
                    print("-" * 40)
            
            elif part == "embeddings":
                print("\n" + "-"*40 + " Running Embeddings Experiments " + "-"*40)
                if not self.kg_embeddings_config:
                    print("Skipping: kg_embeddings_config not provided.")
                    continue
                analysis_embeddings = KGEmbeddings(**self.kg_embeddings_config)
                results = analysis_embeddings.run()
                analysis_embeddings.compare_and_plot_results(results)

            elif part == "prediction":
                print("\n" + "-"*40 + " Running Rent Prediction Model " + "-"*40)
                if not self.kg_prediction_config:
                    print("Skipping: kg_prediction_config not provided.")
                    continue
                analysis_prediction = RentPredictionPipeline(spark=self.spark, **self.kg_prediction_config)
                analysis_prediction.run()