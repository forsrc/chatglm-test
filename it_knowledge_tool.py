from functional_tool import Functional_Tool
from typing import List, Tuple, Any, Union
from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel

class ItKnowledgeTool(Functional_Tool):
    llm: BaseLanguageModel

    # tool description
    name = "软件工程信息查询"
    description = "存有一些软件工程信息的工具，输入应该是对软件工程信息的询问"
    
    # QA params
    context = """
        已知软件工程信息：
        软件工程（英语：software engineering），是软件开发领域里对工程方法的系统应用。
        1968年秋季，NATO（北约）的科技委员会召集了近50名一流的编程人员、计算机科学家和工业界巨头，讨论和制定摆脱“软体危机”的对策。在那次会议上第一次提出了软体工程（software engineering）这个概念，研究和应用如何以系统性的、规范化的、可定量的过程化方法去开发和维护软件，以及如何把经过时间考验而证明正确的管理技术和当前能够得到的最好的技术方法结合起来的学科。它涉及到程序设计语言、数据库、软件开发工具、系统平台、标准、设计模式等方面。其后的几十年里，各种有关软件工程的技术、思想、方法和概念不断被提出，软件工程逐步发展为一门独立的科学。
        1993年，电气电子工程师学会（IEEE）给出了一个更加综合的定义："将系统化的、规范的、可度量的方法用于软件的开发、运行和维护的过程，即将工程化应用于软件开发中"。此后，IEEE多次给出软件工程的定义。
        在现代社会中，软件应用于多个方面。典型的软件比如有电子邮件、嵌入式系统、人机界面、办公套件、操作系统、网页、编译器、数据库、游戏等。同时，各个行业几乎都有计算机软件的应用，比如工业、农业、银行、航空、政府部门等。这些应用促进了经济和社会的发展，提高人们的工作效率，同时提升了生活质量。
        软件工程师是对应用软件创造软件的人们的统称，软件工程师按照所处的领域不同可以分为系统分析师、系统架构师、前端和后端工程师、程序员、测试工程师、用户界面设计师等等。各种软件工程师人们俗称程序员。
 
        名称由来与定义
        软体工程包括两种构面：软体开发技术和软体专案管理。
        软体开发技术：软体开发方法学、软体工具和软体工程环境。
        软体专案管理：软体度量、项目估算、进度控制、人员组织、配置管理、项目计画等。

        软体危机
        主条目：软件危机
        1970年代和1980年代的软体危机。在那个时代，许多软体最后都得到了一个悲惨的结局，软件项目开发时间大大超出了规划的时间表。一些项目导致了财产的流失，甚至某些软件导致了人员伤亡。同时软件开发人员也发现软体开发的难度越来越大。在软体工程界被大量引用的案例是Therac-25的意外：在1985年六月到1987年一月之间，六个已知的医疗事故来自于Therac-25错误地超过剂量，导致患者死亡或严重辐射灼伤[2]。

        由来
        鉴于软体开发时所遭遇困境，北大西洋公约组织（NATO）在1968年举办了首次软体工程学术会议[3]，并于会中提出“软体工程”来界定软体开发所需相关知识，并建议“软体开发应该是类似工程的活动”。软体工程自1968年正式提出至今，这段时间累积了大量的研究成果，广泛地进行大量的技术实践，借由学术界和产业界的共同努力，软体工程正逐渐发展成为一门专业学科。

        定义
        关于软件工程的定义，在GB/T11457-2006《讯息技术 软件工程术语》中将其定义为"应用计算机科学理论和技术以及工程管理原则和方法，按预算和进度，实现满足用户要求的软件产品的定义、开发、和维护的工程或进行研究的学科"。

        包括：
        创立与使用健全的工程原则，以便经济地获得可靠且高效率的软体。[4]
        应用系统化，遵从原则，可被计量的方法来发展、操作及维护软体；也就是把工程应用到软体上。[5]
        与开发、管理及更新软体产品有关的理论、方法及工具。[6]
        一种知识或学科，目标是生产品质良好、准时交货、符合预算，并满足用户所需的软体。[7]
        实际应用科学知识在设计、建构电脑程式，与相伴而来所产生的文件，以及后续的操作和维护上。[8]
        使用与系统化生产和维护软体产品有关之技术与管理的知识，使软体开发与修改可在有限的时间与费用下进行。[9]
        建造由工程师团队所开发之大型软体系统有关的知识学科。[10]
        对软体分析、设计、实施及维护的一种系统化方法。[11]
        系统化地应用工具和技术于开发以计算机为主的应用。[12]
        软体工程是关于设计和开发优质软体。[13]
        软体工程的核心知识（SWEBOK）
        　ACM与IEEE Computer Society联合修定的SWEBOK[14]（Software Engineering Body of Knowledge）提到，软体工程领域中的核心知识包括：

        软体需求（Software requirements）
        软体设计（Software design）
        软体建构（Software construction）
        软体测试（Software test）
        软体维护与更新（Software maintenance）
        软体构型管理（Software Configuration Management, SCM）
        软体工程管理（Software Engineering Management）
        软体开发过程（Software Development Process）
        软体工程工具与方法（Software Engineering Tools and methods）
        软体品质（Software Quality）
            
    """

    qa_template = """
    请根据下面带```分隔符的文本来回答问题。
    如果该文本中没有相关内容可以回答问题，请使用下一个意图回答，如果都没有的话，请直接回复：“抱歉，该问题需要更多上下文信息。
    ”
    ```{text}```
    问题：{query}
    """
    prompt = PromptTemplate.from_template(qa_template)
    llm_chain: LLMChain = None

    def _call_func(self, query) -> str:
        self.get_llm_chain()
        context = self.context
        resp = self.llm_chain.predict(text=context, query=query)
        return resp

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
 