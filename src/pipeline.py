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
        
        # --- Store Configs ---
        self.kg_query_config = kg_query_config
        self.kg_rag_config = kg_rag_config
        self.kg_embeddings_config = kg_embeddings_config
        self.kg_prediction_config = kg_prediction_config
        
        # --- Initialize Stages ---
        self.start_stage = start_stage
        self.max_stage = max_stage
        self.landing = None
        self.formatted = None
        self.trusted = None
        self.exploitation = None
        self.analysis_query = None
        self.analysis_rag = None
        self.analysis_embeddings = None
        self.analysis_prediction = None
        
        # Initialize Spark only if needed for data engineering stages or prediction
        if self.start_stage <= 4 or "prediction" in self.analysis_parts:
            self.spark = get_spark_session()
        else:
            self.spark = None

        # Conditionally initialize pipeline components based on start_stage
        if self.start_stage <= 4: self.exploitation = ExploitationZone(spark=self.spark)
        if self.start_stage <= 3: self.trusted = TrustedZone(spark=self.spark)
        if self.start_stage <= 2: self.formatted = FormattedZone(spark=self.spark)
        if self.start_stage <= 1: self.landing = LandingZone()

        # Conditionally initialize analysis components based on analysis_parts
        if "query" in self.analysis_parts:
            self.analysis_query = KGQueryPipeline(graph_path=Path("./data/exploitation/knowledge_graph.ttl"))

        if "rag_query" in self.analysis_parts:
            if not self.kg_rag_config:
                raise ValueError("kg_rag_config must be provided when 'rag_query' is in analysis_parts.")
            self.analysis_rag = GraphRAGPipeline(
                turtle_path=Path("./data/exploitation/knowledge_graph.ttl"),
                llm_config=self.kg_rag_config.get('llm_config', {})
            )

        if "embeddings" in self.analysis_parts:
            if not self.kg_embeddings_config:
                raise ValueError("kg_embeddings_config must be provided when 'embeddings' is in analysis_parts.")
            self.analysis_embeddings = KGEmbeddings(**self.kg_embeddings_config)

        if "prediction" in self.analysis_parts:
            if not self.kg_prediction_config:
                raise ValueError("kg_prediction_config must be provided when 'prediction' is in analysis_parts.")
            self.analysis_prediction = RentPredictionPipeline(spark=self.spark, **self.kg_prediction_config)


    def run(self):
        """Executes the configured pipeline stages."""
        if self.start_stage <= 1 <= self.max_stage:
            print("\n" + "="*100 + "\nPIPELINE: STARTING LANDING ZONE (STAGE 1)\n" + "="*100)
            self.landing.run()
        
        if self.start_stage <= 2 <= self.max_stage:
            print("\n" + "="*100 + "\nPIPELINE: STARTING FORMATTED ZONE (STAGE 2)\n" + "="*100)
            self.formatted.run()
        
        if self.start_stage <= 3 <= self.max_stage:
            print("\n" + "="*100 + "\nPIPELINE: STARTING TRUSTED ZONE (STAGE 3)\n" + "="*100)
            self.trusted.run()
        
        if self.start_stage <= 4 <= self.max_stage:
            print("\n" + "="*100 + "\nPIPELINE: STARTING EXPLOITATION ZONE (STAGE 4)\n" + "="*100)
            self.exploitation.run()

        if self.start_stage <= 5 <= self.max_stage:
            print("\n" + "="*100 + f"\nPIPELINE: STARTING ANALYSIS (STAGE 5) - Parts: {self.analysis_parts}\n" + "="*100)
            
            # Define the desired execution order for analysis
            analysis_execution_order = ["query", "rag_query", "embeddings", "prediction"]

            for part in analysis_execution_order:
                if part not in self.analysis_parts:
                    continue
                
                # --- Run SPARQL Query Analysis ---
                if part == "query" and self.analysis_query:
                    print("\n" + "-"*40 + " Running SPARQL Query Analysis " + "-"*40)
                    queries_to_run = self.kg_query_config.get('queries', [])
                    if not queries_to_run: print("No queries found in kg_query_config.")
                    for query_info in queries_to_run:
                        print(f"\n--- Executing Query: {query_info.get('name', 'Unnamed Query')} ---")
                        results_df = self.analysis_query.run(sparql_query=query_info['sparql'])
                        print("\n--- Results ---")
                        if results_df.empty:
                            print("No results found for this query.")
                        else:
                            pd.set_option('display.width', 120); pd.set_option('display.max_rows', 100)
                            print(results_df)

                # --- Run GraphRAG Analysis ---
                elif part == "rag_query" and self.analysis_rag:
                    print("\n" + "-"*40 + " Running GraphRAG Analysis " + "-"*40)
                    questions_to_ask = self.kg_rag_config.get('questions', [])
                    if not questions_to_ask: print("No questions found in kg_rag_config.")
                    for q in questions_to_ask:
                        answer = self.analysis_rag.run(q)
                        print("\n< Answer:")
                        print(answer)
                        print("-" * 40)
                
                # --- Run Embeddings Experiments ---
                elif part == "embeddings" and self.analysis_embeddings:
                    print("\n" + "-"*40 + " Running Embeddings Experiments " + "-"*40)
                    results = self.analysis_embeddings.run()
                    self.analysis_embeddings.compare_and_plot_results(results)

                # --- Run Rent Prediction Model ---
                elif part == "prediction" and self.analysis_prediction:
                    print("\n" + "-"*40 + " Running Rent Prediction Model " + "-"*40)
                    self.analysis_prediction.run()

        print("\n" + "="*100 + "\nPIPELINE ENDED\n" + "="*100)
        if self.spark is not None:
            self.spark.stop()
            print("Spark Session stopped.")