from typing import List, Set, Union
from abc import abstractmethod
from functools import total_ordering
from os import path
import os
import pickle
import gc


class Index:
    def __init__(self):
        self.dic_index = {}
        self.set_documents = set()

    def index(self, term: str, doc_id: int, term_freq: int):
        if term not in self.dic_index:
            int_term_id = len(self.dic_index) + 1
            self.dic_index[term] = self.create_index_entry(int_term_id)
        else:
            int_term_id = self.get_term_id(term)

        self.add_index_occur(self.dic_index[term], doc_id, int_term_id, term_freq)
        self.set_documents.add(doc_id)
    

    @property
    def vocabulary(self) -> List[str]:
        return self.dic_index.keys()

    @property
    def document_count(self) -> int:
        return len(self.set_documents)

    @abstractmethod
    def get_term_id(self, term: str):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def create_index_entry(self, termo_id: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def add_index_occur(self, entry_dic_index, doc_id: int, term_id: int, freq_termo: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def get_occurrence_list(self, term: str) -> List:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def document_count_with_term(self, term: str) -> int:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def finish_indexing(self):
        pass

    def write(self, arq_index: str):
        file = open(arq_index, "wb")
        pickle.dump(self, file)
        file.close()
        
    

    @staticmethod
    def read(arq_index: str):
        file = open(arq_index, "rb")
        document = pickle.load(file)
        file.close()
        return document

    def __str__(self):
        arr_index = []
        for str_term in self.vocabulary:
            arr_index.append(f"{str_term} -> {self.get_occurrence_list(str_term)}")

        return "\n".join(arr_index)

    def __repr__(self):
        return str(self)


@total_ordering
class TermOccurrence:
    def __init__(self, doc_id: int, term_id: int, term_freq: int):
        self.doc_id = doc_id
        self.term_id = term_id
        self.term_freq = term_freq

    def write(self, idx_file):
        idx_file.write(self.doc_id.to_bytes(4,byteorder="big"))
        idx_file.write(self.term_id.to_bytes(4,byteorder="big"))
        idx_file.write(self.term_freq.to_bytes(4,byteorder="big"))
        

    def __hash__(self):
        return hash((self.doc_id, self.term_id))

    def __eq__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return False
        return self.term_id == other_occurrence.term_id and self.doc_id == other_occurrence.doc_id

    def __lt__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return False
        return self.term_id < other_occurrence.term_id if self.term_id != other_occurrence.term_id else self.doc_id < other_occurrence.doc_id

    def __gt__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return False
        return self.term_id > other_occurrence.term_id if self.term_id != other_occurrence.term_id else self.doc_id > other_occurrence.doc_id

    def __str__(self):
        return f"( doc: {self.doc_id} term_id:{self.term_id} freq: {self.term_freq})"

    def __repr__(self):
        return str(self)


# HashIndex é subclasse de Index
class HashIndex(Index):
    def get_term_id(self, term: str):
        return self.dic_index[term][0].term_id

    def create_index_entry(self, termo_id: int) -> List:
        return []

    def add_index_occur(self, entry_dic_index: List[TermOccurrence], doc_id: int, term_id: int, term_freq: int):
        entry_dic_index.append(TermOccurrence(doc_id, term_id, term_freq))

    def get_occurrence_list(self, term: str) -> List:
        if term in self.dic_index:
            return self.dic_index[term]
        return []

    def document_count_with_term(self, term: str) -> int:
        return len(self.get_occurrence_list(term))


class TermFilePosition:
    def __init__(self, term_id: int, term_file_start_pos: int = None, doc_count_with_term: int = None):
        self.term_id = term_id

        # a serem definidos após a indexação
        self.term_file_start_pos = term_file_start_pos
        self.doc_count_with_term = doc_count_with_term

    def __str__(self):
        return f"term_id: {self.term_id}, doc_count_with_term: {self.doc_count_with_term}, term_file_start_pos: {self.term_file_start_pos}"

    def __repr__(self):
        return str(self)


class FileIndex(Index):
    TMP_OCCURRENCES_LIMIT = 1000000

    def __init__(self):
        super().__init__()

        self.lst_occurrences_tmp = [None]*FileIndex.TMP_OCCURRENCES_LIMIT
        self.idx_file_counter = 0
        self.str_idx_file_name = None

        # metodos auxiliares para verifica o tamanho da lst_occurrences_tmp
        self.idx_tmp_occur_last_element  = -1
        self.idx_tmp_occur_first_element = 0
        
    def get_tmp_occur_size(self):
        """Retorna o tamanho da lista temporária de ocorrências"""
        return self.idx_tmp_occur_last_element - self.idx_tmp_occur_first_element + 1

    def get_term_id(self, term: str):
        if term in self.dic_index:
            return self.dic_index[term].term_id
        return None

    def create_index_entry(self, term_id: int) -> TermFilePosition:
        return TermFilePosition(term_id)

    def add_index_occur(self, entry_dic_index: TermFilePosition, doc_id: int, term_id: int, term_freq: int):
        #complete aqui adicionando um novo TermOccurrence na lista lst_occurrences_tmp
        #não esqueça de atualizar a(s) variável(is) auxiliares apropriadamente

        term_occurrences = TermOccurrence(doc_id, term_id, term_freq) 
        self.lst_occurrences_tmp[self.idx_tmp_occur_last_element + 1] = term_occurrences
        self.idx_tmp_occur_last_element += 1

        if self.get_tmp_occur_size() >= FileIndex.TMP_OCCURRENCES_LIMIT:
            self.save_tmp_occurrences()

    def next_from_list(self) -> TermOccurrence:
        if self.get_tmp_occur_size() > 0:
            # obtenha o proximo da lista e armazene em next_occur
            # não esqueça de atualizar a(s) variável(is) auxiliares apropriadamente
            next_occur = self.lst_occurrences_tmp[self.idx_tmp_occur_first_element]
            self.idx_tmp_occur_first_element += 1

            return next_occur
        else:
            return None

    def next_from_file(self, file_pointer) -> TermOccurrence:
        # next_from_file = pickle.load(file_idx)
        bytes_doc_id = file_pointer.read(4)
        if not bytes_doc_id:
            return None
        bytes_term_id = file_pointer.read(4)
        if not bytes_term_id:
            return None
        bytes_term_freq = file_pointer.read(4)
        if not bytes_term_freq:
            return None
        
        doc_id = int.from_bytes(bytes_doc_id, byteorder='big')
        term_id = int.from_bytes(bytes_term_id, byteorder='big')
        term_freq = int.from_bytes(bytes_term_freq, byteorder='big')

        return TermOccurrence(doc_id, term_id, term_freq)

    def save_tmp_occurrences(self):

        # Ordena pelo term_id, doc_id
        #    Para eficiência, todo o código deve ser feito com o garbage collector desabilitado gc.disable()
        gc.disable()

        """comparar sempre a primeira posição
        da lista com a primeira posição do arquivo usando os métodos next_from_list e next_from_file
        e use o método write do TermOccurrence para armazenar cada ocorrencia do novo índice ordenado"""

        self.lst_occurrences_tmp = [x for x in self.lst_occurrences_tmp if x is not None]

        if self.str_idx_file_name == None:
            self.str_idx_file_name = "occur_index_{}.idx".format(self.idx_file_counter)
            new_file = open(self.str_idx_file_name, 'wb')
            self.lst_occurrences_tmp.sort()

            next_term_from_list = self.next_from_list()

            while next_term_from_list != None:
                next_term_from_list.write(new_file)
                next_term_from_list = self.next_from_list()

        else:
            current_file = open(self.str_idx_file_name, 'rb')
            self.idx_file_counter = self.idx_file_counter + 1
            self.str_idx_file_name = "occur_index_{}.idx".format(self.idx_file_counter)

            new_file = open(self.str_idx_file_name, 'wb')

            self.lst_occurrences_tmp.sort()
            next_term_from_list = self.next_from_list()
            next_term_from_file = self.next_from_file(current_file)

            while next_term_from_list != None or next_term_from_file != None:
                if next_term_from_file is None:
                    next_term_from_list.write(new_file)
                    next_term_from_list = self.next_from_list()
                elif next_term_from_list is None:
                    next_term_from_file.write(new_file)
                    next_term_from_file = self.next_from_file(current_file)
                elif next_term_from_list < next_term_from_file:
                    next_term_from_list.write(new_file)
                    next_term_from_list = self.next_from_list()
                else:
                    next_term_from_file.write(new_file)
                    next_term_from_file = self.next_from_file(current_file)

            current_file.close()
        
        new_file.close()

        self.lst_occurrences_tmp = [None]*FileIndex.TMP_OCCURRENCES_LIMIT
        self.idx_tmp_occur_last_element = -1
        self.idx_tmp_occur_first_element = 0

        gc.enable()

    def finish_indexing(self):
        if len(self.lst_occurrences_tmp) > 0:
            self.save_tmp_occurrences()

            # Sugestão: faça a navegação e obtenha um mapeamento
            # id_termo -> obj_termo armazene-o em dic_ids_por_termo
        dic_ids_por_termo = {}  # ids sao as chaves e as palavras sao os itens
        for str_term, obj_term in self.dic_index.items():
            dic_ids_por_termo[obj_term.term_id] = str_term

        with open(self.str_idx_file_name, 'rb') as idx_file:
            # Usar o next_from_file pra ir pegando cada registro
            next_term_from_file = self.next_from_file(idx_file)
            seek_file = 0
            while next_term_from_file is not None:
                # pegando chave do dic_index
                str_term = dic_ids_por_termo[next_term_from_file.term_id]

                # Quantidade de vezes que a palavra aparece
                # lembrar que eles ja estao em ordem no arquivo
                qtde = self.dic_index[str_term].doc_count_with_term
                if qtde is None:
                    qtde = 0
                # so ir somando 1
                self.dic_index[str_term].doc_count_with_term = qtde + 1
                # quando aparecer a ultima ocorrencia, vai chegar no valor correto

                # Atualizando a posicao de inicio
                # Tem que atualizar a pos so para a primeira ocorrencia daquela palavra
                if self.dic_index[str_term].term_file_start_pos is None:
                    self.dic_index[str_term].term_file_start_pos = seek_file

                # Chamar o proximo e andar com o seeker do arquivo
                next_term_from_file = self.next_from_file(idx_file)
                # Cada registro tem tamanho 12
                seek_file = seek_file + 12

    def get_occurrence_list(self, term: str) -> List:
        with open(self.str_idx_file_name, 'rb') as idx_file:
            occurrence_list = []
            while True:
                current_term = self.next_from_file(idx_file)
                if current_term is None or self.get_term_id(term) is None:
                    break
                if current_term.term_id != self.get_term_id(term):
                    continue
                occurrence_list.append(current_term)

        return occurrence_list

    def document_count_with_term(self, term: str) -> int:
        if term in self.dic_index:
            return self.dic_index[term].doc_count_with_term
        return 0
