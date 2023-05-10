import datetime
import os
import time

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from config import embedding_model_name


class Document:
    page_content: str = ''
    metadata: dict = {"time": '', 'category': 'Title'}

    def __init__(self, QA: tuple):
        self.page_content = str(QA)
        self.metadata["time"] = datetime.date.today()


class SourceService:
    def __init__(self, vector_store_name):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store_path = './faissCache'
        self.vector_store_name = vector_store_name
        self.vector_store = None

    def init_source_vector(self, docs=None):
        """
        初始化本地Faiss知识库向量
        :return:
        """
        if docs is None:
            docs = [Document(('1', '2')), ]

            # docs = [Document(('今天是几号？', f'今天是{time.strftime("%Y年%m月%d日", time.localtime())}')),]
        if os.path.exists(f'{self.vector_store_path}/{self.vector_store_name}.faiss'):
            # 加载
            print("加载")
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings,
                                                 index_name=self.vector_store_name)
        else:
            # 初始化
            print("初始化")
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(self.vector_store_path, index_name=self.vector_store_name)

    def add_document(self, QA: tuple):
        """
        向Faiss中添加文档
        :param QA: 一次问答
        :return:
        """
        # aa = Document(QA)
        docs = [Document(QA), ]

        self.vector_store.add_documents(docs)
        self.vector_store.save_local(self.vector_store_path, index_name=self.vector_store_name)

    def find_document(self, issue: str, top_k: int = 4):
        """
        搜索文档
        :param top_k: 几个
        :param issue: 问题
        :return: list[Document
        """
        search_result = self.vector_store.similarity_search_with_score(issue, top_k)
        # print(search_result)
        content = ''
        ttext = """{day}天前的对话：{QA} \n"""
        for dd in search_result:
            # print(dd)
            # print(dd[0].page_content)
            # print(type(dd[0]))
            # print(dd[0].page_content)
            # print(dd[0].metadata)
            day = str((datetime.date.today() - dd[0].metadata["time"]).days)
            txt = ttext.format(day=day, QA=dd[0].page_content)
            content = content + txt
        return content

    def find_document_tuple(self, issue: str, top_k: int = 3)->list[tuple]:
        """
        搜索文档
        :param top_k: 几个
        :param issue: 问题
        :return: list[Document
        """
        search_result = self.vector_store.similarity_search_with_score(issue, top_k)
        # print(search_result)
        content = []

        for dd in search_result:
            QA=dd[0].page_content
            QA = eval(QA)
            content.append(QA)

        return content


if __name__ == '__main__':
    aa = SourceService('test1')
    aa.init_source_vector()
    aa.add_document(('从明天起做一个幸福的人', '喂马劈柴周游世界'))
    print(aa.find_document_tuple('今天是几号'))
