# src/pipeline.py

from spark_session import get_spark_session
from landing.landing import LandingZone
from formatted.formatted import FormattedZone
from trusted.trusted import TrustedZone
from exploitation.exploitation import ExploitationZone
from analysis.kg_embeddings import KGEmbeddings
from analysis.kg_query import KGQueryPipeline
from analysis.kg_prediction import RentPredictionPipeline
from typing import List, Dict, Any

class Pipeline:
    """
    Main class to orchestrate the entire data and analysis pipeline.
    """
    def __init__(self, start_stage: int = 1, max_stage: int = 5,
                 analysis_parts: List[str] = ["embeddings", "query", "prediction"],
                 kg_embeddings_config: Dict[str, Any] = None,
                 kg_prediction_config: Dict[str, Any] = None):
        
        # --- Validate Inputs ---
        allowed_parts = {"embeddings", "query", "prediction"}
        if not all(part in allowed_parts for part in analysis_parts):
            raise ValueError(f"Invalid analysis_parts. Allowed values are: {allowed_parts}")
        
        # --- Initialize Stages ---
        self.start_stage = start_stage
        self.max_stage = max_stage
        self.analysis_parts = analysis_parts
        self.landing = None
        self.formatted = None
        self.trusted = None
        self.exploitation = None
        self.analysis_embeddings = None
        self.analysis_prediction = None
        self.analysis_query = None

        # Initialize Spark only if needed for early stages
        if self.start_stage <= 4 or "prediction" in self.analysis_parts:
            self.spark = get_spark_session()
        else:
            self.spark = None

        if self.start_stage <= 4:
            self.exploitation = ExploitationZone(spark=self.spark)

        if self.start_stage <= 3:
            self.trusted = TrustedZone(spark=self.spark)

        if self.start_stage <= 2:
            self.formatted = FormattedZone(spark=self.spark)

        if self.start_stage <= 1:
            self.landing = LandingZone()

        # Conditionally initialize analysis components
        if "embeddings" in self.analysis_parts:
            if not kg_embeddings_config:
                raise ValueError("kg_embeddings_config must be provided when 'embeddings' is in analysis_parts.")
            self.analysis_embeddings = KGEmbeddings(**kg_embeddings_config)

        if "prediction" in self.analysis_parts:
            if not kg_prediction_config:
                raise ValueError("kg_prediction_config must be provided when 'prediction' is in analysis_parts.")
            self.analysis_prediction = RentPredictionPipeline(spark=self.spark, **kg_prediction_config)

        if "query" in self.analysis_parts:
            self.analysis_query = None # Placeholder

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
            
            if self.analysis_embeddings:
                print("\n" + "-"*40 + " Running Embeddings Experiments " + "-"*40)
                results = self.analysis_embeddings.run()
                self.analysis_embeddings.compare_and_plot_results(results)

            if self.analysis_query:
                print("\n" + "-"*40 + " Running SPARQL Query Analysis " + "-"*40)
                # self.analysis_query.run() # To be implemented
                print("SPARQL query part is not yet implemented.")

            if self.analysis_prediction:
                print("\n" + "-"*40 + " Running Rent Prediction Model " + "-"*40)
                self.analysis_prediction.run()

        print("\n" + "="*100 + "\nPIPELINE ENDED\n" + "="*100)
        if self.spark is not None:
            self.spark.stop()
            print("Spark Session stopped.")