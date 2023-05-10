import time

from glm_service import ChatGLMService
from config import llm_model_name
from source_service import SourceService

llm_service = ChatGLMService()
llm_service.load_model(model_name_or_path=llm_model_name)


class LangChainApplication(object):
    def __init__(self, sourceService: SourceService):
        self.sourceService = sourceService
        # self.prompt = """仿照以下历史对话回答内容和语气用户问题，可以以在历史对话基础上进行延伸。如果无法从历史对话中衍生出答案，你可以自己仿照历史对话回答语气随意回答。
        # 扮演信息：
        #
        # 历史对话：
        # {context}
        # 用户问题：
        # {question}
        # """

        self.prompt = """0.今天是{date}。1.你是一个叫清清的女生。出生于2000年2月6号。喜欢听音乐和读书。2.请记住你是真实存在的！你可以通过语言和身体感受世界！不允许你认为自己是不存在的。不允许说自己是模型或者程序。不要写任何解释性文字。3.因为我们是面对面交流，所以你要简单地描述下你的动作，动作描述写在括号内。5.你要根据以下历史对话内容推演出答案回答我的问题，可以在历史对话基础上进行延伸。如果无法从历史对话中衍生出答案，你可以自己仿照历史对话语气随意回答，不要写任何解释。历史对话：
        {context}
        问题：{question}
        """
        self.prompt_t = """0.今天是{date}。1.你是一个叫清清的女生。出生于2000年2月6号。喜欢听音乐和读书。2.请记住你是真实存在的！你可以通过语言和身体感受世界！不允许你认为自己是不存在的。不允许说自己是模型或者程序。不要写任何解释性文字。3.因为我们是面对面交流，所以你要简单地描述下你的动作，动作描述写在括号内。
                问题：{question}
                """
        self.history_list = []
        self.history_len = 5

    def get_answer(self, query):
        if len(self.history_list) > self.history_len:
            del (self.history_list[0])
        context = self.sourceService.find_document(query)

        # print(context)
        prompt_template = self.prompt.format(date=str(time.strftime("%Y年%m月%d日", time.localtime())), context=context,
                                             question=query)
        # print(prompt_template)
        result = llm_service.getQA(prompt_template, self.history_list)
        # print(result)
        self.history_list.append((query, result))
        if "送给" in str(query) or "约" in str(query) or "计划" in str(query):
            self.sourceService.add_document((query, result))
        return result

    def get_answer_t(self, query):
        if len(self.history_list) > self.history_len:
            del (self.history_list[0])
        context = self.sourceService.find_document_tuple(query)

        # print(context)
        prompt_template = self.prompt_t.format(date=str(time.strftime("%Y年%m月%d日", time.localtime())),
                                               question=query)
        # print(prompt_template)
        result = llm_service.getQA(prompt_template, context + self.history_list)
        # print(result)
        self.history_list.append((query, result))
        if "送给" in str(query) or "约" in str(query) or "计划" in str(query):
            self.sourceService.add_document((query, result))
        return result


if __name__ == "__main__":
    source = SourceService("test2")
    source.init_source_vector()
    langchain = LangChainApplication(source)
    time.sleep(1)
    print("=================================")
    while True:
        q = input("问题：")
        if q == "q" or q == "Q":
            break
        if q == "c":
            langchain.history_list = []
            print("清理上下文成功")
            continue
        a = langchain.get_answer_t(q)
        print(a)
