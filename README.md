# GLM_memory
GLM+langchain 长久记忆（尝试阶段）

受AI西部小镇有感
利用faiss 存储用户与模型的对话。并在新对话展开时，基于用户问题而搜索。
先尝试了两种方法：
  将搜索到的对话添加进prompt，
  测试让其记住简单的约定。
  ![图片](https://github.com/ChangAnYX/GLM_memory/assets/126737340/a971caaa-a802-400a-a09f-d4450ed0b170)

  将搜索到的对话添加进history，
  测设效果稍好但也差不多。

不论是因为LLM的聪明程度问题，还是因为记忆方式抑或prompt格式问题
总之效果均不是很理想。

如果想复刻AI西部小镇的话，可能要为每一个虚拟人new一个专属class存储包括人物信息，人物性格，日程规划，拥有物品等等。
并且包括每个虚拟人的偏好，也做为一个重要度考量。
类似于强迫症人格在按时吃饭和与恋人分手之间会选择按时吃饭。。。。

但西部小镇是在一片固定区域，固定人物，固定关系中进行的大模型相关推衍。不清楚能不能实现与用户的长期记忆功能。感觉道路很遥远。

### 如果您有好的思路想法抑或论文，希望您能留言告诉我。
