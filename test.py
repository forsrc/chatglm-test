from langchain.agents import AgentExecutor
from it_knowledge_tool import ItKnowledgeTool
from design_pattern_knowledge_tool import DesignPatternKnowledgeTool
from intent_agent import IntentAgent

from chatglm import ChatGLM

llm = ChatGLM(model_path="HUDM/chatglm2-6b")
llm.load_model()        


tools = [DesignPatternKnowledgeTool(llm=llm), ItKnowledgeTool(llm=llm), ]

agent = IntentAgent(tools=tools, llm=llm)
# agent.choose_tools("中介者模式是什么？")
agent_exec = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=1)
agent_exec.run("中介者模式是什么？")