# exploitation_zone_kg.py

import json
from pathlib import Path
import traceback
import pandas as pd
from tqdm import tqdm

# --- Imports de RDFLib (AMB LA CORRECCIÓ) ---
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD # Importa els namespaces directament

# --- Imports de PySpark ---
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# --- Configuració ---
# Defineix el teu Namespace personalitzat
PROJ = Namespace("http://example.com/catalonia-ontology/")

# Camins a les dades d'entrada
TRUSTED_DIR = Path("./data/trusted/")
RELATIONS_DIR = Path("./data/relations/")
EXPLOITATION_DIR = Path("./data/exploitation/")

# Assegura't que el directori de sortida existeix
EXPLOITATION_DIR.mkdir(parents=True, exist_ok=True)

class KGExploitationZone:
    """
    Orquestra la creació d'un Knowledge Graph integrat a partir de les
    dades de la Trusted Zone i fitxers de relacions geogràfiques.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.graph = self._initialize_graph()
        print("Knowledge Graph Exploitation Zone Initialized.")

    def _initialize_graph(self) -> Graph:
        """Inicialitza un graf RDFLib i l'enllaça amb els prefixos estàndard i personalitzats."""
        g = Graph()
        g.bind("proj", PROJ)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)
        return g

    def _load_data(self) -> dict:
        """Llegeix totes les fonts de dades necessàries (Delta i JSON)."""
        print("\n--- Pas 1: Carregant totes les fonts de dades ---")
        data = {}
        try:
            data['idescat'] = self.spark.read.format("delta").load(str(TRUSTED_DIR / "idescat"))
            data['lloguer'] = self.spark.read.format("delta").load(str(TRUSTED_DIR / "lloguer"))
            data['rfdbc'] = self.spark.read.format("delta").load(str(TRUSTED_DIR / "rfdbc"))
            print("Taules Delta de la Trusted Zone carregades amb èxit.")

            with open(RELATIONS_DIR / "municipality_neighbors.json", 'r') as f:
                data['mun_neighbors'] = json.load(f)
            with open(RELATIONS_DIR / "comarca_neighbors.json", 'r') as f:
                data['com_neighbors'] = json.load(f)
            with open(RELATIONS_DIR / "comarca_to_province.json", 'r') as f:
                data['com_to_prov'] = json.load(f)
            print("Fitxers de relacions JSON carregats amb èxit.")
            return data
        except Exception as e:
            print(f"ERROR: No s'ha pogut carregar una o més fonts de dades. Detall: {e}")
            raise

    def _process_data_for_kg(self, data: dict) -> dict:
        """Processa i transforma els DataFrames de Spark per preparar-los per a la generació de triples."""
        print("\n--- Pas 2: Processant DataFrames de Spark ---")
        
        print("Processant Idescat: trobant l'últim valor per indicador i fent pivot...")
        window_spec = Window.partitionBy("municipality_id", "indicator_id").orderBy(F.col("reference_year").desc())
        df_idescat_latest = data['idescat'].withColumn("rank", F.row_number().over(window_spec)).filter(F.col("rank") == 1).drop("rank")
        df_idescat_pivoted = df_idescat_latest.groupBy("municipality_id", "municipality_name", "comarca_name").pivot("indicator_id").agg(F.first("municipality_value"))
        
        print("Processant Lloguer: agregant anualment...")
        df_lloguer_annual = data['lloguer'].filter(F.col("ambit_territorial") == "Municipi").groupBy("codi_territorial", "any").agg(
            F.avg("renda").alias("avg_monthly_rent"),
            F.sum("habitatges").alias("total_contracts")
        ).withColumnRenamed("codi_territorial", "municipality_id")
        
        print("Processant RFDBC: filtrant per renda per càpita...")
        df_rfdbc_annual = data['rfdbc'].filter(F.col("indicator_code") == "PER_CAPITA_EUR").select(
            F.col("municipality_id"),
            F.col("year").cast("integer").alias("any"),
            F.col("value").alias("household_income")
        )

        print("Unint dades anuals de Lloguer i Ingressos...")
        df_annual_data = df_lloguer_annual.join(
            df_rfdbc_annual,
            ["municipality_id", "any"],
            "full_outer"
        )
        
        return {
            "static_data": df_idescat_pivoted,
            "annual_data": df_annual_data,
            **{k: v for k, v in data.items() if k not in ['idescat', 'lloguer', 'rfdbc']}
        }

    def _populate_graph(self, processed_data: dict):
        """Itera sobre els DataFrames processats i els diccionaris per generar els triples RDF."""
        print("\n--- Pas 3: Poblant el Knowledge Graph amb triples RDF ---")

        static_df = processed_data['static_data']
        annual_df = processed_data['annual_data']
        mun_neighbors = processed_data['mun_neighbors']
        com_neighbors = processed_data['com_neighbors']
        com_to_prov = processed_data['com_to_prov']
        
        created_nodes = set()

        print("Generant nodes estàtics (Municipis, Comarques, Províncies) i les seves propietats...")
        static_data_local = static_df.collect()

        for row in tqdm(static_data_local, desc="Creant Nodes Estàtics"):
            mun_id = row['municipality_id']
            com_name = row['comarca_name']
            
            if not mun_id or not com_name: continue

            mun_uri = PROJ[f"municipality/{mun_id}"]
            com_uri = PROJ[f"comarca/{com_name.replace(' ', '_')}"]
            prov_name = com_to_prov.get(com_name)
            
            if mun_uri not in created_nodes:
                self.graph.add((mun_uri, RDF.type, PROJ.Municipality))
                # Aquí s'utilitza RDFS (importat en majúscules)
                self.graph.add((mun_uri, RDFS.label, Literal(row['municipality_name'], lang='ca')))
                self.graph.add((mun_uri, PROJ.isInComarca, com_uri))
                created_nodes.add(mun_uri)

            if com_uri not in created_nodes:
                self.graph.add((com_uri, RDF.type, PROJ.Comarca))
                self.graph.add((com_uri, RDFS.label, Literal(com_name, lang='ca')))
                if prov_name:
                    prov_uri = PROJ[f"province/{prov_name.replace(' ', '_')}"]
                    self.graph.add((com_uri, PROJ.isInProvince, prov_uri))
                    if prov_uri not in created_nodes:
                        self.graph.add((prov_uri, RDF.type, PROJ.Province))
                        self.graph.add((prov_uri, RDFS.label, Literal(prov_name, lang='ca')))
                        created_nodes.add(prov_uri)
                created_nodes.add(com_uri)

            for col_name in static_df.columns:
                if col_name.startswith('f') and row[col_name] is not None:
                    prop_uri = PROJ[col_name]
                    # Aquí s'utilitza XSD (importat en majúscules)
                    self.graph.add((mun_uri, prop_uri, Literal(row[col_name], datatype=XSD.double)))

        print("Generant relacions de veïnatge...")
        for mun_id, data in mun_neighbors.items():
            mun1_uri = PROJ[f"municipality/{mun_id}"]
            for neighbor in data['neighbors']:
                mun2_uri = PROJ[f"municipality/{neighbor['id']}"]
                self.graph.add((mun1_uri, PROJ.isNeighborOf, mun2_uri))

        for com_name, data in com_neighbors.items():
            com1_uri = PROJ[f"comarca/{com_name.replace(' ', '_')}"]
            for neighbor_name in data['neighbors']:
                com2_uri = PROJ[f"comarca/{neighbor_name.replace(' ', '_')}"]
                self.graph.add((com1_uri, PROJ.isAdjacentTo, com2_uri))

        print("Generant nodes de dades anuals (AnnualDataPoint)...")
        annual_data_local = annual_df.collect()
        for row in tqdm(annual_data_local, desc="Creant Nodes Anuals"):
            mun_id = row['municipality_id']
            year = row['any']
            
            if not mun_id or not year: continue
            
            mun_uri = PROJ[f"municipality/{mun_id}"]
            datapoint_uri = PROJ[f"datapoint/{mun_id}_{year}"]
            
            self.graph.add((datapoint_uri, RDF.type, PROJ.AnnualDataPoint))
            self.graph.add((mun_uri, PROJ.hasAnnualData, datapoint_uri))
            self.graph.add((datapoint_uri, PROJ.referenceYear, Literal(year, datatype=XSD.gYear)))
            
            if row['avg_monthly_rent'] is not None:
                self.graph.add((datapoint_uri, PROJ.avgMonthlyRent, Literal(row['avg_monthly_rent'], datatype=XSD.double)))
            if row['household_income'] is not None:
                self.graph.add((datapoint_uri, PROJ.householdIncome, Literal(row['household_income'], datatype=XSD.double)))

    def run(self):
        """Executa el pipeline complet de la Zona d'Explotació."""
        try:
            raw_data = self._load_data()
            processed_data = self._process_data_for_kg(raw_data)
            self._populate_graph(processed_data)
            
            output_file = EXPLOITATION_DIR / "knowledge_graph.ttl"
            print(f"\n--- Pas 4: Desant el Knowledge Graph a {output_file} ---")
            self.graph.serialize(destination=str(output_file), format='turtle')
            print(f"Graf desat amb èxit. Conté {len(self.graph)} triples.")
            print("\n--- Exploitation Zone (KG) Task Successfully Completed ---")

        except Exception as e:
            print(f"!!! ERROR durant l'execució de la Zona d'Explotació: {e}")
            traceback.print_exc()

# --- Funció per inicialitzar Spark (la que has proporcionat) ---
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.3.0" # Aquesta versió pot variar

def get_spark_session() -> SparkSession:
    """
    Initializes and returns a SparkSession configured for Delta Lake.
    """
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("KG_Exploitation_Pipeline") \
            .master("local[*]") \
            .config("spark.jars.packages", DELTA_PACKAGE) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.parquet.int96AsTimestamp", "true") \
            .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR") # Redueix la verbositat
        print("Spark Session Initialized. Log level set to ERROR.")
        return spark
    except Exception as e:
        print(f"FATAL: Error initializing Spark Session: {e}")
        raise

# --- Bloc Principal d'Execució ---
if __name__ == "__main__":
    spark_session = None
    try:
        spark_session = get_spark_session()
        kg_processor = KGExploitationZone(spark=spark_session)
        kg_processor.run()
    finally:
        if spark_session:
            print("\nAturant Spark Session.")
            spark_session.stop()