"""Main file to run the entire pipeline of the project."""

import os
from dotenv import load_dotenv
from setup import make_folder_structure
from pipeline import Pipeline

if __name__ == "__main__":
	# Load environment variables from a .env file at the project root
	load_dotenv()

	make_folder_structure()

	# ==========================================================================
	# --- PIPELINE & ANALYSIS CONFIGURATIONS ---
	# ==========================================================================

	# --- 1. SPARQL Query Configuration ---
	kg_query_config = {
		'queries': [
			{
				'name': "Debug: Girona Population",
				'sparql': """
					PREFIX proj: <http://example.com/catalonia-ontology/>
					PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
					SELECT ?population
					WHERE {
					  ?municipality rdfs:label "Girona"@ca .
					  ?municipality proj:hasObservation ?pop_obs .
					  ?pop_obs proj:field <http://example.com/catalonia-ontology/indicator/population> ;
							   proj:value ?population .
					}
				"""
			},
			{
				'name': "Debug: Sabadell Annual Data 2021",
				'sparql': """
					PREFIX proj: <http://example.com/catalonia-ontology/>
					PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
					PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
					SELECT ?total_contracts ?income_per_capita
					WHERE {
						?municipality rdfs:label "Sabadell"@ca .
						?municipality proj:hasAnnualData ?data_point .
						?data_point proj:referenceYear "2021"^^xsd:gYear ;
									proj:totalContracts ?total_contracts ;
									proj:incomePerCapita ?income_per_capita .
					}
				"""
			},
			{
				'name': "Debug: El Pont de Suert Neighbors",
				'sparql': """
					PREFIX proj: <http://example.com/catalonia-ontology/>
					PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
					SELECT ?neighbor_label
					WHERE {
					  ?municipality rdfs:label "Pont de Suert, el"@ca .
					  ?municipality proj:isNeighborOf ?neighbor_uri .
					  ?neighbor_uri rdfs:label ?neighbor_label .
					}
				"""
			},
			{
				'name': "Complex: Girona Province Neighbors of Populous Towns with Cheap Rent",
				'sparql': """
					PREFIX proj: <http://example.com/catalonia-ontology/>
					PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
					PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
					SELECT DISTINCT ?municipality_name ?avg_rent ?neighbor_name ?neighbor_population
					WHERE {
					  ?province rdfs:label "Girona"@ca .
					  ?comarca proj:isInProvince ?province .
					  ?mun proj:isInComarca ?comarca ;
						   rdfs:label ?municipality_name .
					  ?mun proj:isNeighborOf ?neighbor_mun .
					  ?neighbor_mun rdfs:label ?neighbor_name .
					  ?neighbor_mun proj:hasObservation ?pop_obs .
					  ?pop_obs proj:field <http://example.com/catalonia-ontology/indicator/population> ;
							   proj:value ?neighbor_population .
					  FILTER(?neighbor_population > 20000)
					  ?mun proj:hasAnnualData ?data_point .
					  ?data_point proj:referenceYear "2021"^^xsd:gYear ;
								  proj:avgMonthlyRent ?avg_rent .
					  FILTER(?avg_rent < 700)
					}
					ORDER BY ?municipality_name
				"""
			}
		]
	}

	# --- 2. Graph RAG (LLM) Query Configuration ---
	kg_rag_config = {
		'llm_config': {
			"model": "gpt-4o",
			"temperature": 0
		},
		'questions': [
			"What is the population of the municipality of Girona?",
			"For the municipality of Sabadell, what was its total number of rental contracts and its income per capita in 2021?",
			"List the municipalities that are neighbors of 'El Pont de Suert' (name)."
		]
	}

	# --- 3. KG Embeddings Training/Evaluation Configuration ---
	kg_embeddings_config = {
		'model_configs': [
			{'name': 'TransH', 'dim': 128},
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
		],
		'epochs': 300, # High because early stopping is implemented
		'batch_size': 64,
		'force_training': False, # Set to True to retrain all models
	}

	# --- 4. Final Rent Prediction Configuration ---
	kg_prediction_config = {
		'best_experiment_id': 'TransH_dim_128',
	}

	# ==========================================================================
	# --- PIPELINE EXECUTION ---
	# ==========================================================================
	# Control which parts of the project to run.
	# start_stage: 1=Landing, 2=Formatted, 3=Trusted, 4=Exploitation, 5=Analysis

	# --- Define which parts of the analysis to run ---
	analysis_parts_to_run = ["query", "rag_query", "embeddings", "prediction"]

	# --- Pre-flight check: Ensure necessary secrets are loaded if required ---
	if "rag_query" in analysis_parts_to_run and not os.getenv("OPENAI_API_KEY"):
		raise ValueError(
			"The 'rag_query' analysis requires an OpenAI API key. "
			"Please create a .env file in the project root with: "
			"OPENAI_API_KEY='your-key-here'"
		)
	
	pipeline = Pipeline(
		start_stage=1,
		max_stage=5,
		analysis_parts=analysis_parts_to_run,
		kg_query_config=kg_query_config,
		kg_rag_config=kg_rag_config,
		kg_embeddings_config=kg_embeddings_config,
		kg_prediction_config=kg_prediction_config
	)
	
	pipeline.run()