from neo4j import GraphDatabase
import pandas as pd
import json

def get_authentication():
    with open('Data/neo4j_info.json') as f:
        neo4j_info = json.load(f)
    f.close()
    return neo4j_info

class Ontology:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=3600*24*30, keep_alive=True)

    def close(self):
        self.driver.close()

    def get_procdure(self, ttp_list):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_procdure, ttp_list) 
            return result

    @staticmethod
    def _return_procdure(tx, ttp_list):
        result = tx.run("MATCH (vo:Verb_Object)-[r]-(t:Technique) RETURN t.id, vo.clean_sentence")
        result =  [ [record['t.id'].split('.')[0], record['vo.clean_sentence']] for record in result]
        
        rtn_result=[]
        for record in result:
            if record[0] in ttp_list:
                rtn_result.append(record[1])

        return rtn_result

    def transID_to_name(self, _id):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_transID_to_name, _id) 
            return result

    @staticmethod
    def _return_transID_to_name(tx,_id):
        result = tx.run("MATCH (vo:Verb_Object {attack_pattern_id:$_id}) RETURN vo.verb_obj",_id=_id)
        result =  [ record['vo.verb_obj'] for record in result][0]
        result = eval(result)
        return result

    def transID_to_tech(self, att_id):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_transID_to_tech, att_id) 
            return result

    @staticmethod
    def _return_transID_to_tech(tx,att_id):
        result = tx.run("MATCH (t:Technique)-[r]-(vo:Verb_Object {attack_pattern_id:$att_id}) RETURN t.id, t.name", att_id=att_id)
        result = [[record['t.id'], record['t.name']] for record in result]
        return result

    def get_tech_description(self):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_tech_description) 
            return result

    @staticmethod
    def _return_tech_description(tx):
        result = tx.run("MATCH (t:Technique) RETURN t.id, t.description")

        result = [ [ record['t.id'], record['t.description'] ] for record in result]
        return result

    def infer_attack_pattern_3(self, g_lst, s_lst, vo_lst, t_lst):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_infer_attack_pattern_3, g_lst, s_lst, vo_lst, t_lst) 
            return result

    @staticmethod
    def _return_infer_attack_pattern_3(tx, g_lst, s_lst, vo_lst, t_lst):
        result_ttp = []
        vo_lst = tx.run("MATCH (vo:Verb_Object)-[r]-(t:Technique) WHERE vo.attack_pattern_id IN $vo_lst RETURN t.id", vo_lst=vo_lst)
        vo_lst = [ record['t.id'].split('.')[0] for record in vo_lst]

        hf_ttp = ['t1047', 't1113', 't1033', 't1123', 't1115', 't1135', 't1082', 't1071', 't1053', 't1106', 't1005', 't1140', 't1036', 't1029', 't1185', 't1566', 't1016', 't1087', 't1059', 't1020', 't1070', 't1049', 't1057', 't1068', 't1095', 't1012', 't1132', 't1485', 't1189', 't1134', 't1136', 't1105', 't1220', 't1008', 't1569', 't1571', 't1505', 't1574', 't1547', 't1564', 't1572', 't1567', 't1518', 't1553', 't1559', 't1560', 't1568', 't1583']

        # in hp_ttp, vo_pair is better
        for ttp in hf_ttp:
            if ttp in vo_lst:
                result_ttp.append(ttp)
            if ttp in t_lst:
                t_lst.remove(ttp)

        result_ttp += t_lst
        return result_ttp

    def infer_attack_pattern_2(self, g_lst, s_lst, t_lst):
        # sub-tech全換tech
        with self.driver.session() as session:
            g_t, s_t, t_t, g_t_t, s_t_t, g_s_t_t = session.write_transaction(self._return_infer_attack_pattern_2, g_lst, s_lst, t_lst) 
            return g_t, s_t, t_t, g_t_t, s_t_t, g_s_t_t

    @staticmethod
    def _return_infer_attack_pattern_2(tx, g_lst, s_lst, t_lst):
        
        g_t = tx.run("MATCH (g:Group)-[r]-(t:Technique) WHERE g.id IN $g_lst RETURN t.id", g_lst=g_lst)
        g_t = [ record['t.id'].split('.')[0] for record in g_t]

        s_t = tx.run("MATCH (s:Software)-[r]-(t:Technique) WHERE s.id IN $s_lst RETURN t.id", s_lst=s_lst)
        s_t = [  record['t.id'].split('.')[0] for record in s_t]
        #print("s_t = ", s_t)

        t_t = t_lst
        t_lst=[]
        # get sub tech, in t_lst
        tech = tx.run("MATCH (t:Technique) RETURN t.id" )
        all_tech  = [ record['t.id'] for record in tech ]
        for t1 in t_t:
            for t2 in all_tech:
                if t1==t2.split('.')[0]:
                    t_lst.append(t2)
        
        g_t_t = list(set(t_t).intersection(g_t))
        s_t_t = list(set(t_t).intersection(s_t))
        g_s_t_t = list(set(g_t_t).intersection(s_t))

        return g_t, s_t, t_t, g_t_t, s_t_t, g_s_t_t

    def infer_attack_pattern(self, g_lst, s_lst, vo_lst):
        """
        result = [ 'level1':{ {'g_t': [ttp1, ...]}, {'s_t':[ttp1, ...], {'vo_t':[ttp1, ...]} },
                   'level2':{ {'g_vo_t':[ttp1, ...]}, {'s_vo_t':[ttp1, ...]} },
                   'level3':{ {'g_s_vo_t':[ttp1, ...]  } }
                 
                 }]
        """ 
        with self.driver.session() as session:
            g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t = session.write_transaction(self._return_infer_attack_pattern, g_lst, s_lst, vo_lst) 
            return g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t
    
    @staticmethod
    def _return_infer_attack_pattern(tx, g_lst, s_lst, vo_lst):
        
        g_t = tx.run("MATCH (g:Group)-[r]-(t:Technique) WHERE g.id IN $g_lst RETURN t.id", g_lst=g_lst)
        g_t = [ [ record['t.id'] ] for record in g_t]
        #print("g_t = ", g_t)

        s_t = tx.run("MATCH (s:Software)-[r]-(t:Technique) WHERE s.id IN $s_lst RETURN t.id", s_lst=s_lst)
        s_t = [ [ record['t.id'] ] for record in s_t]
        #print("s_t = ", s_t)

        vo_t = tx.run("MATCH (vo:Verb_Object)-[r]-(t:Technique) WHERE vo.attack_pattern_id IN $vo_lst RETURN t.id", vo_lst=vo_lst)
        vo_t = [ [ record['t.id'] ] for record in vo_t]
        #print("vo_t = ", vo_t)

        g_vo_t = tx.run("MATCH (g:Group)-[r1]-(vo:Verb_Object)-[r2]-(t:Technique) "
                        "WHERE (g.id IN $g_lst) and (vo.attack_pattern_id IN $vo_lst) RETURN t.id", g_lst=g_lst, vo_lst=vo_lst)
        g_vo_t = [ [ record['t.id'] ] for record in g_vo_t]
        #print("g_vo_t = ", g_vo_t)

        #problem
        s_vo_t = tx.run("MATCH (s:Software)-[r1]-(vo:Verb_Object)-[r2]-(t:Technique) "
                        "WHERE (s.id IN $s_lst) and (vo.attack_pattern_id IN $vo_lst) RETURN t.id", s_lst=s_lst, vo_lst=vo_lst)
        s_vo_t = [ [ record['t.id'] ] for record in s_vo_t]
        #print("s_vo_t = ", s_vo_t)

        g_s_vo_t = tx.run("MATCH (g:Group)-[r1]-(s:Software)-[r2]-(vo:Verb_Object)-[r3]-(t:Technique) "
                          "WHERE (g.id IN $g_lst) and (s.id IN $s_lst) and (vo.attack_pattern_id IN $vo_lst) "
                          "RETURN t.id", g_lst=g_lst, s_lst=s_lst, vo_lst=vo_lst)
        g_s_vo_t = [ [ record['t.id'] ] for record in g_s_vo_t]

        return g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t

    def query_all_group_name(self):
        # result = [ [g1, [alias1,alias2,... ], [g2, [alias1,alias2,... ]   ]
        with self.driver.session() as session:
            result = session.write_transaction(self._return_all_group_name) 
            return result
    @staticmethod
    def _return_all_group_name(tx):
        result = tx.run("MATCH (g:Group) RETURN g.name, g.alias")
        return [ [ record['g.name'], record['g.alias'] ] for record in result]
    
    def query_all_software_name(self):
        # [ [s1, [alias1,alias2,... ], [s2, [alias1,alias2,... ]   ]
        with self.driver.session() as session:
            result = session.write_transaction(self._return_all_software_name) 
            return result
    @staticmethod
    def _return_all_software_name(tx):
        result = tx.run("MATCH (s:Software) RETURN s.name, s.alias")
        s_name_lst = [ [record['s.name'],record['s.alias'] ]  for record in result]
        for index, i in enumerate(s_name_lst):
            # in real name
            if i[0].lower() == 'at':
                s_name_lst[index][0] = 'at.exe'
            # in nickname
            if 'at' in i[1]:
                s_name_lst[index][1].remove('at')
            if 'Page' in i[1]:
                s_name_lst[index][1].remove('Page')
                
        return s_name_lst

    def query_all_match_group(self, group_name_lst):
        # return group id = [g1, g2, ...]
        with self.driver.session() as session:
            result = session.write_transaction(self._return_all_match_group, group_name_lst) 
            return result

    @staticmethod
    def _return_all_match_group(tx, group_name_lst):
        group_name_lst = [i.lower() for i in group_name_lst ]
        result = tx.run("MATCH (g:Group) WHERE toLower(g.name) IN $group_name_lst "
                        "RETURN g.id", group_name_lst=group_name_lst )
        result_lst=[]
        for resord in result:
            result_lst.append(resord['g.id'])
        return result_lst

    def query_all_match_software(self, software_name_lst):
        # return software id = [s1, s2, ...]
        with self.driver.session() as session:
            result = session.write_transaction(self._return_all_match_software, software_name_lst) 
            return result

    @staticmethod
    def _return_all_match_software(tx, software_name_lst):
        software_name_lst = [i.lower() for i in software_name_lst ]
        result = tx.run("MATCH (s:Software) WHERE toLower(s.name) IN $software_name_lst "
                        "RETURN s.id", software_name_lst=software_name_lst )
        result_lst=[]
        for resord in result:
            result_lst.append(resord['s.id'])
        return result_lst

    def query_all_Verb_Object(self):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_Verb_Object) 
            return result

    @staticmethod
    def _return_Verb_Object(tx):
        result = tx.run("MATCH (vo:Verb_Object) "
                        "RETURN vo.attack_pattern_id, vo.verb_obj, vo.sentence ")
        verb_obj_lst=[]
        for record in result:
            verb_obj_lst.append([record["vo.attack_pattern_id"], record["vo.verb_obj"]])
        return verb_obj_lst

    def query_all_procedure_example(self):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_procedure_example)
            return result

    @staticmethod
    def _return_procedure_example(tx):
        result = tx.run("MATCH (vo:Verb_Object)-[r]-(t:Technique) "
                        "RETURN t.id, vo.clean_sentence ")

        pe_list = [ [record['t.id'], record['vo.clean_sentence']]  for record in result]
        return pe_list

    # def query_all_Verb_Object_with_13_Tactic(self):
    #     with self.driver.session() as session:
    #         result = session.write_transaction(self._return_all_Verb_Object_with_13_Tactic)
    #         return result
    # @staticmethod
    # def _return_all_Verb_Object_with_13_Tactic(tx):
    #     result = tx.run("MATCH (vo:Verb_Object)-[r]-(te:Technique) RETURN te.id, vo.attack_pattern_id, vo.verb_obj, vo.srl_label")
    #     result = [ [record['te.id'], record['vo.attack_pattern_id'], record['vo.verb_obj'], record['vo.srl_label']] for record in result ] # [tech, example]

    #     result_df = pd.DataFrame()
    #     tactic_lst=[]
    #     id_lst=[]
    #     vo_lst=[]
    #     srl_lst=[]
    #     for i,row in enumerate(result):
    #         result = tx.run("MATCH (t1:Tactic)-[r]-(t:Technique{id:$_id}) RETURN t1.id", _id=row[0])
    #         tactic_result =  [ record['t1.id'] for record in result ]
    #         # TA0003, TA0004 = TA000304
    #         if ('ta0004' in tactic_result) and ('ta0003' in tactic_result) and (len(tactic_result)==1):
    #             tactic_result = ['ta000304']
    #         if len(tactic_result) >1:
    #             continue

    #         tactic_lst.append(tactic_result[0])
    #         id_lst.append(row[1])
    #         vo_lst.append(eval(row[2]))
    #         srl_lst.append(eval(row[3]))

    #     result_df['tactic'] = tactic_lst
    #     result_df['att_patt'] = id_lst
    #     result_df['vo_pair'] = vo_lst
    #     result_df['srl_label'] = srl_lst

    #     return result_df
    
    def query_all_Verb_Object_with_Tactic(self):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_all_Verb_Object_with_Tactic)
            return result
    @staticmethod
    def _return_all_Verb_Object_with_Tactic(tx):
        result = tx.run("MATCH (vo:Verb_Object)-[r]-(te:Technique) RETURN te.id, vo.attack_pattern_id, vo.verb_obj, vo.srl_label")
        result = [ [record['te.id'], record['vo.attack_pattern_id'], record['vo.verb_obj'], record['vo.srl_label']] for record in result ] # [tech, example]

        result_df = pd.DataFrame()
        tactic_lst=[]
        id_lst=[]
        vo_lst=[]
        srl_lst=[]
        for i,row in enumerate(result):
            _result = tx.run("MATCH (t1:Tactic)-[r]-(t:Technique{id:$_id}) RETURN t1.id", _id=row[0])
            tactic_result =  [ record['t1.id'] for record in _result ]
            tactic_lst.append(tactic_result)
            id_lst.append(row[1])
            vo_lst.append(eval(row[2]))
            srl_lst.append(eval(row[3]))

        result_df['tactic'] = tactic_lst
        result_df['att_patt'] = id_lst
        result_df['vo_pair'] = vo_lst
        result_df['srl_label'] = srl_lst

        return result_df

    def get_all_tech_id_name(self):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_all_tech_id_name)
            return result
    @staticmethod
    def _return_all_tech_id_name(tx):
        result = tx.run("MATCH (t:Technique) WHERE t.sub_tech=$f RETURN t.id, t.name ", f=False)
        result = [ [record['t.id'], record['t.name']] for record in result ]
        result_dict = dict()

        for row in result:
            result_dict[row[0]] = row[1]
        return result_dict
    
    def get_technique2tactic_dict(self):
        with self.driver.session() as session:
            result = session.write_transaction(self._return_technique2tactic_dict)
            return result
    @staticmethod
    def _return_technique2tactic_dict(tx):
        result = tx.run("MATCH (ta:Tactic)-[r]-(t:Technique) RETURN ta.id, t.id ")
        tech2ta_list = [ [record['ta.id'], record['t.id']] for record in result ]
        
        tech2ta_dict=dict()
        for items in tech2ta_list:
            ta=items[0]
            tech=items[1].split('.')[0]

            if tech2ta_dict.get(tech, None)==None:
                tech2ta_dict[tech]= [ta]
            else:
                tech2ta_dict[tech].append(ta)

        for tech, ta_list in tech2ta_dict.items():
            tech2ta_dict[tech] = list(set(ta_list))
            
        return tech2ta_dict

if __name__=='__main__':
    neo4j_info = get_authentication()
    greeter = Ontology(neo4j_info["url"],neo4j_info["account"], neo4j_info["password"])
    x = greeter.get_technique2tactic_dict()
    greeter.close()
    print(x)
