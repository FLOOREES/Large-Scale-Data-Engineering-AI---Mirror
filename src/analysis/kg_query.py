from pathlib import Path
import pandas as pd
from rdflib import Graph

class KGQueryPipeline:
    """
    A professional and minimalist pipeline to load and query a Knowledge Graph.

    This class is designed to be lean. It initializes by loading a graph
    from a file and provides a clean interface to execute SPARQL queries,
    returning the results in a user-friendly format (a Pandas DataFrame).
    """

    def __init__(self, graph_path: Path):
        """
        Initializes the query pipeline by loading the Knowledge Graph.

        Args:
            graph_path (Path): The file path to the RDF graph (e.g., a .ttl file).
        """
        print("KG Query Pipeline firing up! Let's find some answers.")
        self.graph = self._load_graph(graph_path)

    def _load_graph(self, graph_path: Path) -> Graph:
        """
        Loads an RDF graph from a file into an rdflib.Graph object.
        
        Think of this step as opening the book. We're loading all the facts
        from our file into memory so we can ask questions about them.
        """
        if not graph_path.exists():
            print(f"ERROR: Graph file not found at: {graph_path}")
            raise FileNotFoundError(f"The specified graph file does not exist: {graph_path}")

        print(f"Loading graph from '{graph_path}'... this might take a moment.")
        g = Graph()
        # We explicitly tell it the format is 'turtle' (.ttl) which is faster
        # than letting rdflib guess.
        g.parse(source=str(graph_path), format="turtle")
        print(f"Graph loaded successfully! It contains {len(g)} facts (triples).")
        return g

    def run_query(self, sparql_query: str) -> pd.DataFrame: 
        """
        Executes a SPARQL query against the loaded graph and returns a DataFrame.

        This is where the magic happens. We pass our question (the SPARQL query)
        to the graph and get back the answers.

        Args:
            sparql_query (str): A string containing the SPARQL query.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the query results,
                          with columns named after the SPARQL variables.
        """
        print("\nExecuting query...")
        
        # The graph.query() method does the heavy lifting.
        query_results = self.graph.query(sparql_query)
        
        # The results are raw, so let's make them beautiful.
        # We'll transform them into a list of dictionaries.
        records = []
        for row in query_results:
            # For each row of results, create a dictionary where the keys
            # are the variable names from the SPARQL query (like ?name)
            # and the values are the actual results.
            record = {str(var): val.toPython() for var, val in row.asdict().items()}
            records.append(record)

        print(f"Query executed. Found {len(records)} results.")
        
        # A Pandas DataFrame is the most professional and useful way
        # to present tabular results.
        return pd.DataFrame(records)

# --- Minimal Test Block ---
if __name__ == "__main__":
    
    # 1. SETUP
    # Define the path to the graph we created in the exploitation zone.
    graph_file = Path("./data/exploitation/knowledge_graph.ttl")

    # 2. DEFINE A MEANINGFUL QUERY
    # Let's write a question in SPARQL. We want to find the top 5 most
    # populated municipalities in the 'Barcelonès' comarca and see their
    # household income for the year 2021.
    
    # This is a great example because it joins static data (name, population, comarca)
    # with time-series data (income for a specific year).
    query = """
    PREFIX proj: <http://example.com/catalonia-ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?municipality_name ?population ?income
    WHERE {
      ?comarca rdfs:label "Barcelonès"@ca .
      
      ?mun proj:isInComarca ?comarca ;
           rdfs:label ?municipality_name ;
           proj:f171 ?population . # 'f171' is our property for population

      ?mun proj:hasAnnualData ?data_point .
      
      ?data_point proj:referenceYear "2021"^^xsd:gYear ;
                  proj:householdIncome ?income .
    }
    ORDER BY DESC(?population)
    LIMIT 5
    """

    # 3. EXECUTION
    # Let's create our pipeline and run the query!
    try:
        query_pipeline = KGQueryPipeline(graph_path=graph_file)
        results_df = query_pipeline.run_query(sparql_query=query)

        # 4. DISPLAY RESULTS
        # Print the final, beautiful DataFrame.
        print("\n--- Query Results ---")
        if results_df.empty:
            print("No results found for this query.")
        else:
            # Set display options for a cleaner printout in the console.
            pd.set_option('display.width', 100)
            pd.set_option('display.max_columns', 10)
            print(results_df)

    except FileNotFoundError as e:
        print(f"\nCould not run test. Please ensure the graph file exists at '{graph_file}'")
        print("You may need to run the 'exploitation_zone_kg.py' script first.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")