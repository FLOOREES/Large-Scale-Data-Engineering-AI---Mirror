import os
from pathlib import Path

from rdflib import Graph
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class GraphRAGPipeline:
    """
    A professional, modular pipeline for performing GraphRAG.

    This version manually implements the text-to-SPARQL and answer synthesis
    steps using core LLMChains for maximum reliability and control, bypassing
    the higher-level GraphSparqlQAChain wrapper.
    """

    def __init__(self, config: dict):
        """Initializes the GraphRAG pipeline with a given configuration."""
        self.config = config
        self.graph = self._load_rdflib_graph()
        self.llm = self._initialize_llm()
        
        # We now create two separate, explicit chains.
        self.sparql_generation_chain = self._create_sparql_generation_chain()
        self.answer_synthesis_chain = self._create_answer_synthesis_chain()
        
        print("GraphRAG Pipeline Initialized Successfully.")

    def _load_rdflib_graph(self) -> Graph:
        """Loads the RDF graph directly into an rdflib.Graph object."""
        graph_path = self.config["graph_path"]
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found at: {graph_path}")
        
        print(f"Loading graph from '{graph_path}' into rdflib...")
        graph = Graph()
        graph.parse(source=str(graph_path), format="turtle")
        print(f"Graph loaded successfully with {len(graph)} triples.")
        return graph

    def _get_schema(self) -> str:
        """A simple method to extract a text representation of the graph schema."""
        query = """
        SELECT DISTINCT ?class ?predicate ?object
        WHERE {
          ?subject a ?class.
          ?subject ?predicate ?object.
        } LIMIT 100
        """
        # This is a simplified schema extraction. For a production system,
        # this would be more robust, parsing RDFS/OWL definitions.
        results = self.graph.query(query)
        return "\n".join(str(row) for row in results)

    def _initialize_llm(self) -> ChatOpenAI:
        """Initializes the Language Model from the configuration."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return ChatOpenAI(
            temperature=self.config['temperature'],
            model=self.config['llm_model_name']
        )

    def _create_sparql_generation_chain(self) -> LLMChain:
        """Creates the chain responsible for generating SPARQL queries."""
        print("Initializing SPARQL Generation Chain...")
        prompt_template = """
        You are an expert SPARQL developer. Your task is to write a SPARQL query to answer a user's question about a knowledge graph.
        Base the query on the provided graph schema and the question. Return only the SPARQL query code block.

        Graph Schema:
        {schema}

        Question:
        {question}

        SPARQL Query:
        """
        prompt = PromptTemplate(
            input_variables=["schema", "question"], template=prompt_template
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def _create_answer_synthesis_chain(self) -> LLMChain:
        """Creates the chain responsible for generating the final answer."""
        print("Initializing Answer Synthesis Chain...")
        prompt_template = """
        You are a helpful assistant. Given a user's question and the results of a database query,
        synthesize a final, human-readable answer based ONLY on the provided context.

        Question:
        {question}

        Query Results:
        {query_results}

        Final Answer:
        """
        prompt = PromptTemplate(
            input_variables=["question", "query_results"], template=prompt_template
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def run(self, question: str) -> str:
        """
        Executes the full RAG pipeline for a given natural language question.
        """
        print(f"\n--- Processing Question: '{question}' ---")
        
        # Step 1: Generate the SPARQL query.
        print("Step 1: Generating SPARQL query...")
        graph_schema = self._get_schema()
        generated_sparql = self.sparql_generation_chain.run(
            schema=graph_schema, question=question
        )
        # Clean up the LLM's output to get only the code.
        generated_sparql = generated_sparql.strip().replace("```sparql", "").replace("```", "").strip()
        print(f"Generated SPARQL:\n{generated_sparql}")

        # Step 2: Execute the query against our graph.
        print("\nStep 2: Executing query against the graph...")
        try:
            query_results = self.graph.query(generated_sparql)
            results_str = "\n".join(str(row.asdict()) for row in query_results)
            print(f"Query returned {len(query_results)} results.")
        except Exception as e:
            return f"Error executing generated SPARQL query: {e}"

        # Step 3: Synthesize the final answer.
        print("\nStep 3: Synthesizing final answer...")
        final_answer = self.answer_synthesis_chain.run(
            question=question, query_results=results_str
        )
        return final_answer


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("="*60 + "\nERROR: OPENAI_API_KEY environment variable is not set.\n" + "="*60)
    else:
        CONFIG = {
            "graph_path": Path("./data/exploitation/knowledge_graph.ttl"),
            "llm_model_name": "gpt-4",
            "temperature": 0.0,
        }
        rag_pipeline = GraphRAGPipeline(config=CONFIG)
        
        question = "Which are the top 3 most populated municipalities in the Barcelonès comarca?"
        
        final_answer = rag_pipeline.run(question=question)
        print("\n--- Final Answer ---")
        print(final_answer)