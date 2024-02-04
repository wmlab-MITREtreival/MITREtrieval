from neo4j import GraphDatabase
import json

def get_authentication():
    with open('Data/neo4j_info.json') as f:
        neo4j_info = json.load(f)
    f.close()
    return neo4j_info
    
class Ontology:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query_tech_by_tactic(self, tactic):
        with self.driver.session() as session:
            tech = session.write_transaction(self._return_tech, tactic)    
            for i in tech:
                print(i)
            print("len(tech) = ", len(tech))

    @staticmethod
    def _return_tech(tx, tactic):
        result = tx.run("MATCH (n:Technique)-[r:`accomplishes`]-(t:Tactic) "
                        "WHERE t.name=$tactic "
                        "RETURN n.name ", tactic=tactic)
        tech=[]
        for record in result:
            tech.append(record["n.name"])
        return tech
    
    """
    @staticmethod
    def _return_tech(tx, tactic):
        result = tx.run("MATCH (n:technique)-[r:`accomplishes`]-(t:tactic) "
                        "WHERE t.name=$tactic "
                        "RETURN n.name ", tactic=tactic)
        tech=[]
        for record in result:
            tech.append(record["n.name"])
        return tech
    """

if __name__ == "__main__":
    neo4j_info = get_authentication()
    greeter = Ontology(neo4j_info["url"],neo4j_info["account"], neo4j_info["password"])
    #greeter = Ontology("bolt://127.0.0.1", "neo4j", "wmlab")
    greeter.query_tech_by_tactic("Initial Access")
    greeter.close()