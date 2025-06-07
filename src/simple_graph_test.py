from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
import owlrl

# Create a graph
g = Graph()

# Define a namespace
EX = Namespace("http://example.org/")

# Add triples to the graph
g.add((EX.Person, RDF.type, RDFS.Class))
g.add((EX.has_name, RDF.type, RDF.Property))
g.add((EX.has_name, RDFS.domain, EX.Person))
g.add((EX.has_name, RDFS.range, RDFS.Literal))

person1 = URIRef("http://example.org/person1")
person2 = URIRef("http://example.org/person2")

g.add((person1, RDF.type, EX.Person))
g.add((person1, EX.has_name, Literal('Alice')))
g.add((person2, EX.has_name, Literal('Bob')))

# Apply OWL RL reasoning
owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)

# Query the graph after inference
query = """
PREFIX ex: <http://example.org/>
SELECT ?subject
WHERE {
    ?subject a ex:Person.
}
"""

results = g.query(query)
for row in results:
    print(str(row[0]).split("#")[-1])