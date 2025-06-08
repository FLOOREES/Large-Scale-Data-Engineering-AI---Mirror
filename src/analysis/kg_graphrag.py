# --- Imports ---
import os
import re
from pathlib import Path
from typing import Any, Dict

# --- LangChain & RDFLib Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rdflib import Graph as RdfLibGraph


# ==============================================================================
# --- Utility Function: LLM Output ---
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

    def _load_graph(self, turtle_path: Path) -> RdfLibGraph:
        """Loads the graph into a raw rdflib.Graph object for reliable querying."""
        if not turtle_path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found at: {turtle_path}")
        print(f"Loading raw rdflib.Graph from: {turtle_path}")
        graph = RdfLibGraph()
        graph.parse(str(turtle_path), format="turtle")
        print(f"Raw graph loaded. Contains {len(graph)} triples.")
        return graph

    def _create_llm(self, llm_config: Dict[str, Any]) -> ChatOpenAI:
        """Initializes the Chat LLM from configuration."""
        print(f"Creating LLM with config: {llm_config}")
        return ChatOpenAI(**llm_config)

    def _create_rag_chain(self) -> Runnable:
        """
        Creates the full RAG chain using LCEL with a complete, hardcoded schema.
        """
        print("Creating custom RAG chain with authoritative, hardcoded schema prompt...")

        # This is the Grand Unified, Authoritative Prompt. It contains a perfect
        # description of the graph, leaving nothing to chance.
        sparql_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an expert at converting user questions into SPARQL queries for a specific knowledge graph about Catalonia.

# Authoritative Graph Schema
The graph has the following structure and predicates. You MUST adhere to this schema.

## Core Entities
- `proj:Municipality`: A municipality.
- `proj:Comarca`: A comarca (county).
- `proj:Province`: A province.
- `proj:IndicatorObservation`: A data point for a single, non-annual indicator (like population).
- `proj:AnnualDataPoint`: A collection of data points for a specific municipality and year.

## Relationships & Properties

### `proj:Municipality` properties:
- `rdfs:label`: The name of the municipality (e.g., "Girona"@ca, "Pont de Suert, el"@ca).
- `proj:isInComarca`: Links a Municipality to its `proj:Comarca`.
- `proj:isNeighborOf`: Links a Municipality to a neighboring `proj:Municipality`.
- `proj:hasObservation`: Links to an `proj:IndicatorObservation` node.
- `proj:hasAnnualData`: Links to an `proj:AnnualDataPoint` node.

### `proj:AnnualDataPoint` properties:
- `proj:referenceYear`: The year for this data (e.g., "2021"^^xsd:gYear).
- `proj:avgMonthlyRent`: The average monthly rent.
- `proj:totalContracts`: The total number of rental contracts.
- `proj:incomePerCapita`: The income per capita in EUR.
- `proj:incomeIndex`: The income index compared to Catalonia=100.
- `proj:incomeTotal`: The total income in thousands of EUR.

### `proj:IndicatorObservation` properties:
- `proj:field`: A direct URI link to the indicator type. Use the full URI. (e.g., `<http://example.com/catalonia-ontology/indicator/population>`).
- `proj:value`: The numeric value of the observation.

# Querying Rules
1.  You MUST include `PREFIX` declarations for `proj`, `rdfs`, and `xsd` at the start of every query.
2.  When matching a `rdfs:label` for a geographical entity, you MUST use the Catalan language tag `@ca`.
3.  For municipalities with articles (el, la, l'), the official format is `Name, article` and the article MUST BE LOWERCASE (e.g., "Pont de Suert, el"@ca).

# Your Turn
Generate a SPARQL query for the following question.

Question: {question}
SPARQLQuery:
""",
        )

        sparql_generation_chain = (
            sparql_prompt
            | self.llm
            | StrOutputParser()
            | _extract_sparql_query
        )
        
        qa_prompt = PromptTemplate.from_template(
            """You are an assistant that answers user questions based on provided context.
If the context is empty or contains no results, say you do not have sufficient information for that query.
Your answer should be clear, concise, and directly based on the information given.

Context:
{context}

Question: {question}
Answer:"""
        )

        def run_sparql_and_get_context(sparql_query: str) -> Dict[str, Any]:
            print("\n--- Generated SPARQL ---\n", sparql_query)
            try:
                query_results = self.graph.query(sparql_query)
                return {"context": str([r.asdict() for r in query_results])}
            except Exception as e:
                return {"context": f"SPARQL query failed with error: {e}"}

        full_rag_chain = (
            # The input to this whole chain is a dictionary: {"question": "..."}
            RunnablePassthrough.assign(sparql_query=sparql_generation_chain)
            | RunnablePassthrough.assign(
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
        "model": "gpt-4o",
        "temperature": 0
    }

    try:
        rag_pipeline = GraphRAGPipeline(
            turtle_path=GRAPH_FILE_PATH,
            llm_config=LLM_CONFIGURATION
        )

        questions_to_ask = [
            "What is the population of the municipality of Girona?",
            "For the municipality of Sabadell, what was its total number of rental contracts and its income per capita in 2021?",
            "List the municipalities that are neighbors of 'El Pont de Suert' (name)."
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