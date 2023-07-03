from typing import List, Set, Mapping
from nltk.tokenize import word_tokenize
from util.time import CheckTime
from query.ranking_models import RankingModel, VectorRankingModel, IndexPreComputedVals, BooleanRankingModel, OPERATOR
from index.structure import Index, TermOccurrence
from index.indexer import Cleaner


class QueryRunner:
    def __init__(self, ranking_model: RankingModel, index: Index, cleaner: Cleaner):
        self.ranking_model = ranking_model
        self.index = index
        self.cleaner = cleaner

    def get_relevance_per_query(self) -> Mapping[str, Set[int]]:
        """
        Adiciona a lista de documentos relevantes para um determinada query (os documentos relevantes foram
        fornecidos no ".dat" correspondente. Por ex, belo_horizonte.dat possui os documentos relevantes da consulta "Belo Horizonte"

        """
        dic_relevance_docs = {}
        for arquiv in ["belo_horizonte", "irlanda", "sao_paulo"]:
            with open(f"relevant_docs/{arquiv}.dat") as arq:
                dic_relevance_docs[arquiv] = set(int(s) for s in arq.readline().split(","))
        return dic_relevance_docs

    def count_topn_relevant(self, n: int, respostas: List[int], doc_relevantes: Set[int]) -> int:
        """
        Calcula a quantidade de documentos relevantes na top n posições da lista lstResposta que é a resposta a uma consulta
        Considere que respostas já é a lista de respostas ordenadas por um método de processamento de consulta (BM25, Modelo vetorial).
        Os documentos relevantes estão no parametro docRelevantes
        """

        # print(f"Respostas: {respostas} doc_relevantes: {doc_relevantes}")
        relevance_count = 0

        position = 0
        while position < n and position < len(respostas):
            if respostas[position] in doc_relevantes:
                relevance_count += 1
            position += 1

        return relevance_count

    def compute_precision_recall(self, n: int, lst_docs: List[int], relevant_docs: Set[int]) -> (float, float):

        precision = None
        recall = None
        return precision, recall

    def get_query_term_occurence(self, query: str) -> Mapping[str, TermOccurrence]:
        """
            Preprocesse a consulta da mesma forma que foi preprocessado o texto do documento (use a classe Cleaner para isso).
            E transforme a consulta em um dicionario em que a chave é o termo que ocorreu
            e o valor é uma instancia da classe TermOccurrence (feita no trabalho prático passado).
            Coloque o docId como None.
            Caso o termo nao exista no indic, ele será desconsiderado.
        """
        map_term_occur = {}
        dic_word_count = {}

        tokens = word_tokenize(query.lower(), "portuguese")

        for token in tokens:
            if token is not None:
                if token not in dic_word_count:
                    dic_word_count[token] = 0
                dic_word_count[token] += 1

        for term, count in dic_word_count.items():
            if self.index.get_occurrence_list(term):
                map_term_occur[term] = TermOccurrence(None, self.index.get_term_id(term), count)

        return map_term_occur

    def get_occurrence_list_per_term(self, terms: List) -> Mapping[str, List[TermOccurrence]]:
        """
            Retorna dicionario a lista de ocorrencia no indice de cada termo passado como parametro.
            Caso o termo nao exista, este termo possuirá uma lista vazia
        """
        dic_terms = {}

        query = " ".join(terms)

        tokens = word_tokenize(query.lower(), "portuguese")

        for term in tokens:
            term = self.cleaner.preprocess_word(term)
            dic_terms[term] = self.index.get_occurrence_list(term)

        return dic_terms

    def get_docs_term(self, query: str) -> List[int]:
        """
            A partir do indice, retorna a lista de ids de documentos desta consulta
            usando o modelo especificado pelo atributo ranking_model
        """

        # Obtenha, para cada termo da consulta, sua ocorrencia por meio do método get_query_term_occurence
        dic_query_occur = self.get_query_term_occurence(query)

        # obtenha a lista de ocorrencia dos termos da consulta
        dic_occur_per_term_query = self.get_occurrence_list_per_term(query.split("_"))

        # utilize o ranking_model para retornar o documentos ordenados considrando dic_query_occur e dic_occur_per_term_query
        return self.ranking_model.get_ordered_docs(query=dic_query_occur, docs_occur_per_term=dic_occur_per_term_query)

    @staticmethod
    def runQuery(query: str, indice: Index, indice_pre_computado: IndexPreComputedVals,
                 map_relevantes: Mapping[str, Set[int]]):
        """
            Para um daterminada consulta `query` é extraído do indice `index` os documentos mais relevantes, considerando
            um modelo informado pelo usuário. O `indice_pre_computado` possui valores précalculados que auxiliarão na tarefa.
            Além disso, para algumas consultas, é impresso a precisão e revocação nos top 5, 10, 20 e 50. Essas consultas estão
            Especificadas em `map_relevantes` em que a chave é a consulta e o valor é o conjunto de ids de documentos relevantes
            para esta consulta.
        """
        time_checker = CheckTime()

        cleaner = Cleaner(stop_words_file="stopwords.txt",
                          language="portuguese",
                          perform_stop_words_removal=True,
                          perform_accents_removal=True,
                          perform_stemming=False)

        # PEça para usuario selecionar entre Booleano ou modelo vetorial para intanciar o QueryRunner
        # apropriadamente. NO caso do booleano, vc deve pedir ao usuario se será um "and" ou "or" entre os termos.
        # Abaixo, existem exemplos fixos
        option_model = input('Qual modelo?\nOpções: \nb -> Booleano\nv -> Vetorial')
        option_ranking_model_bool = 'o'
        if option_model == 'b':
            option_ranking_model_bool = input('Qual operação booleana?\nOpções: \n a -> AND\n o -> OR')

        if option_model == 'v':
            ranking_model = VectorRankingModel(indice_pre_computado)
        elif option_ranking_model_bool == 'o':
            ranking_model = BooleanRankingModel(OPERATOR.OR)
        else:
            ranking_model = BooleanRankingModel(OPERATOR.AND)

        qr = QueryRunner(indice, ranking_model=ranking_model, cleaner=cleaner)
        time_checker.print_delta("Query Creation")

        # Utilize o método get_docs_term para obter a lista de documentos que responde esta consulta
        resposta = qr.get_docs_term(query)
        if option_model == 'v':
            resposta = list(resposta)
        time_checker.print_delta(f"anwered with {len(resposta)} docs")

        # nesse if, vc irá verificar se o termo possui documentos relevantes associados a ele
        # se possuir, vc deverá calcular a Precisao e revocação nos top 5, 10, 20, 50.
        # O for que fiz abaixo é só uma sugestao e o metododo countTopNRelevants podera auxiliar no calculo da revocacao e precisao
        if (query in map_relevantes):
            relevants_list = list(map_relevantes[query])
            if any(relevant_doc in resposta for relevant_doc in relevants_list):
                arr_top = [5, 10, 20, 50]
                revocacao = 0
                precisao = 0
                doc_relevante = map_relevantes[query]
                for n in arr_top:
                    precisao, revocacao = qr.compute_precision_recall(n, resposta, doc_relevante)
                    print(f"Precisao @{n}: {precisao}")
                    print(f"Recall @{n}: {revocacao}")

        # imprima aas top 10 respostas
        top_respostas = resposta[:10]
        print("Top 10 melhores respostas:")
        for i, item in enumerate(top_respostas):
            print(f"{i + 1}: {item}")

    @staticmethod
    def main():
		##TODO
        # leia o indice (base da dados fornecida)
        index = None

        # Checagem se existe um documento (apenas para teste, deveria existir)
        print(f"Existe o doc? index.hasDocId(105047)")

        # Instancie o IndicePreCompModelo para pr ecomputar os valores necessarios para a query
        print("Precomputando valores atraves do indice...");
        check_time = CheckTime()

        check_time.print_delta("Precomputou valores")

        # encontra os docs relevantes
        map_relevance = None

        print("Fazendo query...")
        # aquui, peça para o usuário uma query (voce pode deixar isso num while ou fazer um interface grafica se estiver bastante animado ;)
        query = "São Paulo";
        QueryRunner.runQuery(query, idx, idxPreCom, mapRelevances);
