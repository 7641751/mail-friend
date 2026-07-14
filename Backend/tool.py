from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain.agents import AgentState

from langgraph.types import Command
# 加载环境变量
from dotenv import load_dotenv

"""
工具函数
"""


load_dotenv()
class AuthenticatedState(AgentState):
    authenticated: bool

# 定义工具，用于用户邮箱鉴权
@tool(description="Authenticate user email with password")
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    # 定义变量，记录校验结果
    authenticated = False
    message = "Authentication failed"
    # 校验邮箱和密码
    if email == "huge@itcast.cn" and password == "123":
        authenticated = True
        message = "Successfully authenticated"

    # 返回校验结果
    return Command(
            update={
                "authenticated": authenticated,
                "messages": [
                    ToolMessage(message, tool_call_id=runtime.tool_call_id)
                ],
            }
        )


#%%
# 定义工具，收取和发送邮件
@tool
def check_inbox() -> str:
    """Read an email from the given address."""
    # 模拟收件箱邮件
    return [
        {
            "subject": "周末见个面？",
            "content": """
                嗨 虎哥，
                我下周会去城里，不知道我们有没有机会一起喝杯咖啡？

                祝好，简
            """,
            "from": "jane@itcast.cn",
            "status": "unread"
        },
        {
            "subject": "周五会议",
            "content": """
                嗨 虎哥，
                非常抱歉，我周五的会议无法准时参加了，能不能重新安排个时间？

                祝好，小李
            """,
            "from": "lixiaolong@itcast.cn",
            "status": "checked"
        }
    ]


@tool
def send_email(to: str, subject: str, body: str) -> str:
    # 模拟发送邮件
    """Send an response email"""
    return f"邮件已发送至 {to} , 主题： {subject} , 内容： {body}"
#%%
from typing import Callable
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
# 定义动态工具调用，根据授权状态决定允许的工具
@wrap_model_call
def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Allow read inbox and send email tools only if user provides correct email and password"""

    # 读取授权状态
    authenticated = request.state.get("authenticated")

    if authenticated:
        tools = [check_inbox, send_email]
    else:
        tools = [authenticate]

    # 重写工具列表
    request = request.override(tools=tools)
    return handler(request)
from langchain.agents.middleware import dynamic_prompt

# 授权前，要求鉴定用户权限的系统提示词
unauthenticated_prompt = """You are a helpful email assistant.
    For system security protocols, you must authenticate user before any other interaction.
    """

# 授权后，邮件处理的系统提示词
authenticated_prompt = "You are a helpful assistant that can check the inbox and send emails."

@dynamic_prompt
def dynamic_prompt_fn(request: ModelRequest) -> str:
    """Use different prompt based on whether user is authenticated."""
    authenticated = request.state.get("authenticated")
    if authenticated:
        return authenticated_prompt
    else:
        return unauthenticated_prompt
