from typing import List
from abc import abstractmethod
from typing import List, Set,Mapping
from index.structure import TermOccurrence
import math
from enum import Enum

class IndexPreComputedVals():
    def __init__(self,index):
        self.index = index
        self.precompute_vals()

    def precompute_vals(self):
        """
        Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """
        self.document_norm = {}
        self.doc_count = self.index.document_count

        for i in range(self.doc_count):
            sum = 0
            for term, index_occur in self.index.dic_index.items():
                lst_occurrence = self.index.get_occurrence_list(term)
                for term_occurrence in lst_occurrence:
                    if term_occurrence.doc_id == (i+1):
                        w = VectorRankingModel.tf_idf(self.doc_count, term_occurrence.term_freq, index_occur.doc_count_with_term)
                        sum += math.pow(w,2)
            self.document_norm[i+1] = math.sqrt(sum)
        
class RankingModel():
    @abstractmethod
    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              docs_occur_per_term:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def rank_document_ids(self,documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key= lambda x:-documents_weight[x])
        return doc_ids

class OPERATOR(Enum):
  AND = 1
  OR = 2
    
#Atividade 1
class BooleanRankingModel(RankingModel):
    def __init__(self,operator:OPERATOR):
        self.operator = operator

    def intersection_all(self,map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> List[int]:
        set_ids = set()
        first_iteration = True
        for term, lst_occurrences in map_lst_occurrences.items():
            docs_ids = [term_occurrence.doc_id for term_occurrence in lst_occurrences]

            if first_iteration:
                set_ids = set(docs_ids)
                first_iteration = False
            else:
                set_ids = set_ids.intersection(docs_ids)

        return list(set_ids)
    
    def union_all(self,map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> List[int]:
        set_ids = set()
        
        for  term, lst_occurrences in map_lst_occurrences.items():
            docs_ids = []
            for term_occurrence in lst_occurrences: 
                docs_ids.append(term_occurrence.doc_id)
            set_ids = set_ids.union(docs_ids)

        return set_ids

    def get_ordered_docs(self,query:Mapping[str,TermOccurrence], map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        """Considere que map_lst_occurrences possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(map_lst_occurrences),None
        else:
            return self.union_all(map_lst_occurrences),None

#Atividade 2
class VectorRankingModel(RankingModel):

    def __init__(self,idx_pre_comp_vals:IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term:int) -> float:
        return 1 + math.log2(freq_term)

    @staticmethod
    def idf(doc_count:int, num_docs_with_term:int )->float:
        return math.log2(doc_count / num_docs_with_term)

    @staticmethod
    def tf_idf(doc_count:int, freq_term:int, num_docs_with_term) -> float:
        tf = VectorRankingModel.tf(freq_term)
        idf = VectorRankingModel.idf(doc_count, num_docs_with_term)
        #print(f"TF:{tf} IDF:{idf} n_i: {num_docs_with_term} N: {doc_count}")
        return tf*idf

    def get_ordered_docs(self, query:Mapping[str,TermOccurrence], docs_occur_per_term:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        documents_weight = {}
        for term, term_occurrence in docs_occur_per_term.items():
            if term in query:
                w_query = self.tf_idf(self.idx_pre_comp_vals.doc_count, query[term].term_freq, len(term_occurrence))

            for occurrence in term_occurrence:
                w_doc = self.tf_idf(self.idx_pre_comp_vals.doc_count, occurrence.term_freq, len(term_occurrence))

                weight = (w_doc * w_query) / self.idx_pre_comp_vals.document_norm[occurrence.doc_id]
                documents_weight[occurrence.doc_id] = weight if occurrence.doc_id not in documents_weight else documents_weight[occurrence.doc_id] + weight

        #retona a lista de doc ids ordenados de acordo com o TF IDF
        return self.rank_document_ids(documents_weight), documents_weight

