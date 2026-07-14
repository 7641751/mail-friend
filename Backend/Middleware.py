from typing import Callable
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from tool import check_inbox, send_email, authenticate
from langchain.agents.middleware import dynamic_prompt


"""
This file contains middleware for the email agent.
"""

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
