# --- Imports ---
from pathlib import Path
import pandas as pd
from kg_query import KGQueryPipeline

# --- Main Execution Block ---
if __name__ == "__main__":
    
    graph_file = Path("./data/exploitation/knowledge_graph.ttl")

    # This query finds municipalities in Girona province that are neighbors of a
    # populous municipality (>20k) and had cheap rent (<700) in 2021.
    complex_query = """
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
    
    # --- Execute the Query using the Pipeline ---
    try:
        # Initialize your pipeline from the other script
        query_pipeline = KGQueryPipeline(graph_path=graph_file)

        print(f"\n{'='*10} Executing Complex Query {'='*10}")
        results_df = query_pipeline.run_query(sparql_query=complex_query)

        print("\n--- Query Results ---")
        if results_df.empty:
            print("No municipalities found that match all the complex criteria.")
        else:
            pd.set_option('display.width', 120)
            pd.set_option('display.max_columns', 10)
            pd.set_option('display.max_rows', 100)
            print(results_df)

    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not run test. Please ensure the KG file exists at '{graph_file}'")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")