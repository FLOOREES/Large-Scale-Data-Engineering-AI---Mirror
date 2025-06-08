# --- Imports ---
from pathlib import Path
import pandas as pd
from kg_query import KGQueryPipeline

# --- Main Execution Block ---
if __name__ == "__main__":
    
    graph_file = Path("./data/exploitation/knowledge_graph.ttl")

    # --- Step 1: Define the Hand-Crafted "Ground Truth" Queries ---

    # Query 1: Girona Population
    girona_query = """
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

    # Query 2: Sabadell Annual Data for 2021
    sabadell_query = """
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

    # Query 3: El Pont de Suert Neighbors
    pont_de_suert_query = """
    PREFIX proj: <http://example.com/catalonia-ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?municipality ?neighbor_uri ?neighbor_label
    WHERE {
      # 1. Find the starting municipality by its official name.
      ?municipality rdfs:label "Pont de Suert, el"@ca .
      
      # 2. Find all URIs it is connected to via the 'isNeighborOf' predicate.
      ?municipality proj:isNeighborOf ?neighbor_uri .
      
      # 3. OPTIONALLY, try to find the label for the neighbor's URI.
      # If a neighbor_uri shows up but neighbor_label is blank,
      # it means that neighbor node was never fully created in the graph.
      OPTIONAL {
        ?neighbor_uri rdfs:label ?neighbor_label .
      }
    }
    """
    
    # We'll store our questions and queries together for nice, clean output.
    test_suite = {
        "What is the population of the municipality of Girona?": girona_query,
        "For the municipality of Sabadell, what was its total number of rental contracts and its income per capita in 2021?": sabadell_query,
        "List the municipalities that are neighbors of 'El Pont de Suert'.": pont_de_suert_query,
    }

    # --- Step 2: Run the Test Suite ---
    try:
        query_pipeline = KGQueryPipeline(graph_path=graph_file)

        for question, query in test_suite.items():
            print(f"\n{'='*10} Answering: '{question}' {'='*10}")
            results_df = query_pipeline.run(sparql_query=query)

            print("\n--- Results ---")
            if results_df.empty:
                print("No results found in the knowledge graph for this query.")
            else:
                pd.set_option('display.width', 120)
                pd.set_option('display.max_rows', 100)
                print(results_df)

    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not run test. Please ensure the KG file exists at '{graph_file}'")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")