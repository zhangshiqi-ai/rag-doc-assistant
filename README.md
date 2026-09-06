# 本地文档 RAG 问答助手

一个纯本地运行的文档问答小工具：上传 Word / PDF / Markdown / TXT，提出问题，模型基于文档内容回答并标注引用来源。不联网上传你的文档，检索和生成都在本地流程内完成。

## 功能

- 支持上传 `.txt` / `.md` / `.docx` / `.pdf` 四种格式
- 中文文档检索使用本地 bge 模型，无需调用境外 embedding 接口
- 回答带 `[1][2]` 引用编号，对应检索到的原文片段
- 文档中不存在的内容，模型会如实回答"未在文档中找到相关内容"，不编造

## 技术栈

| 环节 | 选型 | 说明 |
|---|---|---|
| 应用框架 | LangChain 1.0+ | 编排加载、切分、检索、生成链路 |
| 向量库 | Chroma | 本地持久化向量存储 |
| 中文检索模型 | BAAI/bge-small-zh-v1.5 | 本地 embedding，免费、不泄露数据 |
| 生成模型 | DeepSeek (deepseek-chat) | 通过 API 调用，负责最终作答 |
| 界面 | Gradio | 浏览器交互 |

## 运行方式

### 1. 环境准备

Python 3.11（已验证兼容），建议新建虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate        # macOS / Linux
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置密钥

复制 `.env.example` 为 `.env`，填入 DeepSeek API Key：

```bash
DEEPSEEK_API_KEY=sk-你的真实key
```

### 4. 启动

由于 bge 模型默认从 Hugging Face 下载，国内建议先设置镜像：

```bash
$env:HF_ENDPOINT="https://hf-mirror.com"   # PowerShell
python app.py
```

浏览器打开 `http://127.0.0.1:7860`，上传文档、输入问题即可。

## 工作流程

```
上传文档
  └─ 按后缀选择加载器（LOADER_MAP）读取文本
       └─ RecursiveCharacterTextSplitter 切分为带重叠的 chunk
            └─ bge 模型向量化，存入 Chroma
                 └─ 用户提问 → 检索 Top-K 相关 chunk
                      └─ 拼接进提示词（含 {context} 占位符）
                           └─ DeepSeek 生成带引用的回答
```

## 关键参数

| 参数 | 位置 | 作用 |
|---|---|---|
| `chunk_size` / `chunk_overlap` | app.py 第 55 行 | 控制文本切片大小与重叠区，太小会切断语义，太大会稀释字段 |
| `k`（检索数量） | app.py 第 59 行 | 每次取前 K 个相关 chunk，太小易漏答案，太大易引入噪声 |
| 提示词 `{context}` | app.py 第 25–32 行 | 必须保留占位符，否则检索到的文档无法传入模型 |

## 目录结构

```
rag-doc-assistant/
├── app.py            # 主程序
├── requirements.txt  # 依赖
├── .env.example      # 密钥模板
└── 逐行精解.md        # 逐行讲解，便于理解每条语句的设计意图
```

## 说明

本项目用于学习 RAG（检索增强生成）的完整链路：数据加载 → 切分 → 向量化 → 检索 → 提示词约束 → 生成。当前为单人演示形态，每次提问重新构建向量库；如需多人使用，可将向量库改为一次构建、持久化复用。
