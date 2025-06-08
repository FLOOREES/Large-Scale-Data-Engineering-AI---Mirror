from pathlib import Path
import pandas as pd
from rdflib import Graph

class KGQueryPipeline:
    """
    A professional and minimalist pipeline to load and query a Knowledge Graph.
    This version contains the corrected SPARQL query.
    """

    def __init__(self, graph_path: Path):
        """
        Initializes the query pipeline by loading the Knowledge Graph.
        """
        print("KG Query Pipeline Initialized.")
        self.graph = self._load_graph(graph_path)

    def _load_graph(self, graph_path: Path) -> Graph:
        """Loads an RDF graph from a file into an rdflib.Graph object."""
        if not graph_path.exists():
            print(f"ERROR: Graph file not found at: {graph_path}")
            raise FileNotFoundError(f"The specified graph file does not exist: {graph_path}")

        print(f"Loading graph from '{graph_path}'...")
        g = Graph()
        g.parse(source=str(graph_path), format="turtle")
        print(f"Graph loaded successfully! It contains {len(g)} facts (triples).")
        return g

    def run_query(self, sparql_query: str) -> pd.DataFrame:
        """
        Executes a SPARQL query and returns the results in a Pandas DataFrame.
        """
        print("\nExecuting SPARQL query...")
        
        query_results = self.graph.query(sparql_query)
        
        records = []
        for row in query_results:
            record = {str(var): val.toPython() for var, val in row.asdict().items()}
            records.append(record)

        print(f"Query executed. Found {len(records)} results.")
        return pd.DataFrame(records)

# --- Main execution block ---
if __name__ == "__main__":
    
    graph_file = Path("./data/exploitation/knowledge_graph.ttl")

    # --- CORRECTED SPARQL QUERY ---
    # The change is on the line identifying the population indicator.
    query = """
    PREFIX proj: <http://example.com/catalonia-ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?municipality_name ?population ?income
    WHERE {
      ?comarca rdfs:label "Barcelonès"@ca .
      ?mun proj:isInComarca ?comarca ;
           rdfs:label ?municipality_name .

      ?mun proj:hasObservation ?pop_obs .
      
      ?pop_obs proj:field <http://example.com/catalonia-ontology/indicator/population> ;
               proj:value ?population .

      ?mun proj:hasAnnualData ?data_point .
      ?data_point proj:referenceYear "2021"^^xsd:gYear ;
                  proj:incomePerCapita ?income .
    }
    ORDER BY DESC(?population)
    LIMIT 5
    """

    try:
        query_pipeline = KGQueryPipeline(graph_path=graph_file)
        results_df = query_pipeline.run_query(sparql_query=query)

        print("\n--- Query Results ---")
        if results_df.empty:
            print("No results found for this query.")
        else:
            pd.set_option('display.width', 100)
            pd.set_option('display.max_columns', 10)
            print(results_df)

    except FileNotFoundError as e:
        print(f"\nCould not run test. Please ensure the graph file exists at '{graph_file}'")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
