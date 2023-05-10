from transformers import AutoModel, AutoTokenizer


class ChatGLMService:
    max_token: int = 100000
    temperature: float = 0.7
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    def load_model(self,
                   model_name_or_path: str = "/app/THUDM/chatglm-6b"):
        """
        构造模型
        :param model_name_or_path:
        :return:
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def getQA(self, prompt: str, history: list[tuple]) -> str:
        """
        问答
        :param prompt:问题
        :param history:历史
        :return:
        """
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        # print(response)
        return response
