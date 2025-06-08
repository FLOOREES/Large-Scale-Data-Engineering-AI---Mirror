# ==============================================================================
#
#                            Graph RAG Pipeline
#
# ==============================================================================
#
# Author: Your Programming Buddy
# Date: 2024-05-21 (Pure LCEL Version)
# Description: A professional, modular pipeline built from scratch using modern
#              LangChain Expression Language (LCEL). This approach offers full
#              transparency and control, avoiding the complexities of legacy
#              chains.
#
# ==============================================================================


# --- Imports ---
import os
import re
from pathlib import Path
from typing import Any, Dict, List

# --- LangChain Core Imports ---
from langchain_community.graphs import RdfGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# --- Utility Function: LLM Output Sanitizer ---
# ==============================================================================

def _extract_sparql_query(text: str) -> str:
    """Extracts the SPARQL query from the LLM's raw output."""
    match = re.search(r"```(sparql\s*)?(.*)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return text.strip()


# ==============================================================================
# --- Class Definition: GraphRAGPipeline ---
# ==============================================================================

class GraphRAGPipeline:
    """
    Orchestrates a Retrieval-Augmented Generation (RAG) pipeline over a
    knowledge graph using SPARQL, built entirely with modern LCEL.
    """

    def __init__(self, turtle_path: Path, llm_config: Dict[str, Any]):
        """Initializes the pipeline by loading the graph, LLM, and the QA chain."""
        print("--- Initializing Graph RAG Pipeline ---")
        self.graph = self._load_graph(turtle_path)
        self.llm = self._create_llm(llm_config)
        self.chain = self._create_rag_chain()
        print("--- Pipeline Initialized Successfully ---")

    # --- Private Helper Methods ---

    def _load_graph(self, turtle_path: Path) -> RdfGraph:
        """Loads the RDF graph from the specified Turtle file."""
        if not turtle_path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found at: {turtle_path}")

        print(f"Loading graph from: {turtle_path}")
        graph = RdfGraph(source_file=str(turtle_path), standard="rdf")
        print(f"Graph loaded. Contains {len(graph.graph)} triples.")
        return graph

    def _create_llm(self, llm_config: Dict[str, Any]) -> ChatOpenAI:
        """Initializes the Chat LLM from configuration."""
        print(f"Creating LLM with config: {llm_config}")
        return ChatOpenAI(**llm_config)

    def _create_rag_chain(self) -> Runnable:
        """
        Creates the full RAG chain using LCEL.
        This involves generating a SPARQL query, executing it, and synthesizing an answer.
        """
        print("Creating custom RAG chain with LCEL...")

        # --- 1. SPARQL Generation Chain ---
        # This chain takes a question and generates a sanitized SPARQL query.
        sparql_prompt = PromptTemplate(
            input_variables=["question", "schema"],
            template="""You are an expert at converting user questions into SPARQL queries.
Given an input question, create a syntactically correct SPARQL query to run.
You must use the provided schema to generate the query. Do not add any text outside of the query itself.

Schema:
{schema}

Question: {question}
SPARQLQuery:
""",
        )

        sparql_generation_chain = (
            RunnablePassthrough.assign(schema=lambda _: self.graph.get_schema)
            | sparql_prompt
            | self.llm
            | StrOutputParser()
            | _extract_sparql_query
        )

        # --- 2. Full RAG Chain ---
        # This orchestrates the entire process.
        qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an assistant that answers user questions based on provided context.
If the context is empty, say you do not have that information.
Your answer should be clear, concise, and directly based on the information given.

Context:
{context}

Question: {question}
Answer:
""",
        )

        # We define a function to run the SPARQL query and format the results.
        def run_sparql_and_get_context(sparql_query: str) -> Dict[str, Any]:
            print("\n--- Generated SPARQL ---\n", sparql_query)
            query_results = self.graph.query(sparql_query)
            return {"context": str([r.asdict() for r in query_results])}

        full_rag_chain = (
            # The input to this whole chain is a dictionary: {"question": "..."}
            RunnablePassthrough.assign(sparql_query=sparql_generation_chain)
            | RunnablePassthrough.assign(
                # The 'context' is derived from running the 'sparql_query'
                context=lambda x: run_sparql_and_get_context(x["sparql_query"])["context"]
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        return full_rag_chain

    # --- Public Interface ---

    def ask(self, question: str) -> str:
        """Asks a question to the RAG pipeline."""
        print(f"\n> Executing query for question: '{question}'")
        # The chain now expects a dictionary with a 'question' key.
        result = self.chain.invoke({"question": question})
        return result


# ==============================================================================
# --- Main Execution Block (for demonstration) ---
# ==============================================================================

if __name__ == "__main__":

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    GRAPH_FILE_PATH = Path("./data/exploitation/knowledge_graph.ttl")

    LLM_CONFIGURATION = {
        "model": "gpt-4o-mini",
        "temperature": 0
    }

    try:
        rag_pipeline = GraphRAGPipeline(
            turtle_path=GRAPH_FILE_PATH,
            llm_config=LLM_CONFIGURATION
        )

        questions_to_ask = [
            "What is the population of the municipality of Girona?",
        ]

        for q in questions_to_ask:
            answer = rag_pipeline.ask(q)
            print("\n< Answer:")
            print(answer)
            print("-" * 40)

    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not start the pipeline. {e}")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")