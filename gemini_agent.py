from vertexai import generative_models
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
from google.cloud import aiplatform
import math

# Step 1: 初始化 Vertex AI 项目与区域
aiplatform.init(
    project="vertex-ai-test-465220",
    location="us-central1"  # Gemini 支持区域
)

# Step 2: 构建一个安全的计算器工具
def calculator_tool(expression: str) -> str:
    try:
        # 限定 eval 的作用域，防止恶意执行
        result = eval(expression, {"__builtins__": None}, {
            "sqrt": math.sqrt,
            "pow": pow,
            "abs": abs,
            "round": round,
            "max": max,
            "min": min
        })
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Step 3: 定义 FunctionDeclaration，告诉 Gemini 工具格式
calculator_function = FunctionDeclaration(
    name="calculator",
    description="Evaluates math expressions like '2+2', 'sqrt(16)', or 'pow(2,3)'.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A valid math expression in Python syntax"
            }
        },
        "required": ["expression"]
    }
)

# Step 4: 注册 Tool
tool = Tool(function_declarations=[calculator_function])

# Step 5: 初始化 Gemini 模型（开启工具支持）
model = GenerativeModel(
    model_name="gemini-1.5-pro-preview-0409",
    tools=[tool]
)

# Step 6: 启动对话 session
chat = model.start_chat()

# Step 7: 输入用户请求
user_prompt = "What is the square root of 256 plus 13 multiplied by 2?"

# 发送第一条消息（自然语言问题）
response = chat.send_message(user_prompt)

# Step 8: 检查 Gemini 是否请求调用工具
called_tool = False

for candidate in response.candidates:
    parts = candidate.content.parts
    if parts and hasattr(parts[0], "function_call"):
        call = parts[0].function_call
        if call.name == "calculator":
            called_tool = True
            expr = call.args.get("expression")
            print(f"🤖 Gemini decided to call calculator with expression: {expr}")

            # Step 9: 本地调用 calculator 工具
            result = calculator_tool(expr)
            print(f"🧮 Calculator result: {result}")

            # Step 10: 把结果拼接成文本回传给 Gemini（代替 FunctionResponse）
            final_reply = chat.send_message(
                f"The result of the calculation ({expr}) is: {result}"
            )
            print("🧠 Final Gemini reply:", final_reply.text)

if not called_tool:
    print("Gemini did not trigger any function call.")
    print("💬 Response:", response.text)
