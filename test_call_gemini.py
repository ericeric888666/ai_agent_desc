#!/usr/bin/env python
# coding: utf-8

# # 初始化 Vertex AI 环境

# In[ ]:


# !pip install -U google-cloud-aiplatform


# In[18]:


from google.cloud import aiplatform


# In[19]:


aiplatform.init(
    project="vertex-ai-test-465220",
    location="us-central1"  # Gemini 目前主要支持这个区域
)


# # 调用 Gemini 模型 API（文本生成示例）

# In[21]:


from vertexai.generative_models import GenerativeModel

# 创建 Gemini 模型实例（可选 gemini-pro 或 gemini-1.5-pro）
model = GenerativeModel("gemini-2.5-pro")

# 调用模型
response = model.generate_content("请用中文总结一下人工智能的主要应用领域。")

# 输出返回内容
print(response.text)


# 将结果写入文件
with open("/gcs/ml-bucket_test-1/output.txt", "w", encoding="utf-8") as f:
    f.write(response.text)


