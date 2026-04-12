# RAG + Tool Calling 模块技术交付文档

> **模块负责人**: Liu Fayang  
> **项目**: AI Shopper Agent — CS5260 Group 30  
> **日期**: 2026-04-12  
> **测试状态**: 15/15 端到端测试全部通过

---

## 目录

1. [模块概述](#1-模块概述)
2. [项目结构](#2-项目结构)
3. [环境搭建与运行](#3-环境搭建与运行)
4. [数据层详解](#4-数据层详解)
5. [索引构建详解](#5-索引构建详解)
6. [三层级检索系统](#6-三层级检索系统)
7. [13个确定性工具](#7-13个确定性工具)
8. [RAG Pipeline 主管道](#8-rag-pipeline-主管道)
9. [集成指南 — 如何对接 Agent 编排层](#9-集成指南--如何对接-agent-编排层)
10. [集成指南 — 如何对接前端](#10-集成指南--如何对接前端)
11. [测试结果](#11-测试结果)
12. [性能与成本分析](#12-性能与成本分析)
13. [已知限制与后续优化](#13-已知限制与后续优化)

---

## 1. 模块概述

### 1.1 职责边界

本模块覆盖 AI Shopper Agent 的中下游：从对话 Agent 输出的改写 query 开始，到返回结构化的推荐结果为止。

```
对话 Agent (上游)          本模块                                  前端 (下游)
┌────────────┐     ┌──────────────────────────────┐     ┌──────────────┐
│ 用户意图解析 │     │  改写query                    │     │              │
│ 多轮追问    │ ──→ │  ↓                            │ ──→ │ 商品卡片      │
│ query 改写  │     │  三层级语义检索 (ChromaDB)      │     │ 对比表       │
│            │     │  ↓                            │     │ 推荐文本      │
│            │     │  13个确定性 Tool 链式过滤       │     │ 下单确认      │
│            │     │  ↓                            │     │              │
│            │     │  PipelineResult (结构化输出)    │     │              │
└────────────┘     └──────────────────────────────┘     └──────────────┘
```

### 1.2 设计原则

- **真实数据**: 从 HuggingFace 加载 Amazon Products 2023 数据集，6784 条真实消费电子商品
- **层级检索**: 品类 → 品牌 → 语义相似度，三层逐级下钻
- **确定性工具**: 所有过滤、比较、排序、统计都通过确定性函数执行，不依赖 LLM 推断
- **零 API 成本**: 检索和工具层完全本地运行（本地 embedding + ChromaDB + SQLite），不消耗 LLM token

---

## 2. 项目结构

```
Proj/
├── data/
│   ├── __init__.py
│   ├── load_dataset.py        # 从 HuggingFace 下载+清洗数据
│   ├── build_index.py         # 构建 ChromaDB + SQLite 双索引
│   ├── sample_products.py     # (旧版) 示例数据，保留兼容
│   ├── products_clean.json    # [生成] 清洗后的数据缓存 (~6784条)
│   ├── chroma_db/             # [生成] ChromaDB 持久化目录
│   └── products.db            # [生成] SQLite 数据库
├── rag/
│   ├── __init__.py
│   ├── retriever.py           # 三层级检索器
│   └── pipeline.py            # RAG 主管道 (唯一入口)
├── agent/
│   ├── __init__.py
│   └── tools.py               # 13个确定性工具函数
├── scripts/
│   └── test_pipeline.py       # 15个端到端测试
├── docs/
│   └── RAG_and_Tools_Technical_Document.md  # 本文档
└── requirements.txt
```

### 2.1 文件依赖关系

```
data/load_dataset.py          ← HuggingFace datasets 库
        ↓
data/build_index.py           ← chromadb, sentence-transformers
        ↓ (生成 chroma_db/ 和 products.db)
rag/retriever.py              ← chromadb (查询)
agent/tools.py                ← sqlite3  (查询)
        ↓
rag/pipeline.py               ← retriever + tools (编排)
        ↓
scripts/test_pipeline.py      ← pipeline  (验证)
```

---

## 3. 环境搭建与运行

### 3.1 依赖安装

```bash
pip install chromadb sentence-transformers datasets
```

| 包 | 用途 | 版本要求 |
|---|---|---|
| `chromadb` | 向量数据库，存储和检索产品 embedding | 最新稳定版 |
| `sentence-transformers` | 本地 embedding 模型 `all-MiniLM-L6-v2` | 最新稳定版 |
| `datasets` | 从 HuggingFace Hub 下载 Amazon 数据集 | 最新稳定版 |

**注意**: 不需要 `langchain`、`openai` 等依赖。本模块完全本地运行，不消耗任何 LLM API token。

### 3.2 首次运行（三步）

```bash
# Step 1: 下载并清洗数据集 (~90秒，需要网络)
python -m data.load_dataset

# Step 2: 构建索引 (~4分钟，本地计算 embedding)
python -m data.build_index --cache

# Step 3: 运行端到端测试验证
python -m scripts.test_pipeline
```

### 3.3 后续运行

索引构建完成后，`data/chroma_db/` 和 `data/products.db` 会持久化到磁盘。后续使用时不需要重新构建，直接 import 即可：

```python
from rag.pipeline import search
result = search("wireless headphones")
```

### 3.4 Windows 编码注意事项

在 Windows 上运行时，建议加 `-X utf8` 参数：

```bash
python -X utf8 -m data.load_dataset
python -X utf8 -m data.build_index --cache
python -X utf8 -m scripts.test_pipeline
```

---

## 4. 数据层详解

### 4.1 数据来源

**数据集**: `iguzelofficial/AMAZON-Products-2023` (HuggingFace)  
**原始规模**: 117,243 条商品，覆盖全品类  
**筛选策略**: 只保留 5 个消费电子相关品类，每个品类上限 2000 条

| 品类 | 筛选后数量 | 品牌数 | 平均评分 |
|---|---|---|---|
| Cell Phones & Accessories | 2,000 | 967 | 4.03 |
| All Electronics | 2,000 | 1,200 | 4.00 |
| Computers | 1,724 | 537 | 4.15 |
| Camera & Photo | 842 | 496 | 4.19 |
| Home Audio & Theater | 218 | 127 | 3.98 |
| **合计** | **6,784** | - | - |

### 4.2 数据清洗流程

`data/load_dataset.py` 对原始数据执行以下清洗步骤：

1. **品类过滤**: 只保留 `TARGET_CATEGORIES` 中的 5 个品类
2. **标题校验**: 丢弃空标题或过短标题 (< 5字符)
3. **价格标准化**: 转为 float，无效值设为 None
4. **品牌提取**: 依次尝试 store 字段、details 中的 Brand/Manufacturer 字段
5. **型号提取**: 从 details 中读取 Item model number / Model Number
6. **子类别提取**: 取 categories 列表的最后一个元素（最具体的分类）
7. **规格提取**: 从 details dict 中筛选硬件相关字段（Battery、Screen Size、RAM 等 30+ 个 key）
8. **描述合并**: 合并 description + features 字段，截断到 2000 字符

### 4.3 清洗后的数据结构

每条商品标准化为以下 dict：

```python
{
    "id": "B0BXYQJDJ5",                    # Amazon ASIN
    "title": "Sony WH-1000XM4 ...",         # 商品标题
    "brand": "Sony",                        # 品牌
    "model": "WH-1000XM4",                  # 型号
    "price": 348.00,                        # 价格 (float | None)
    "main_category": "All Electronics",     # 一级品类
    "subcategory": "Over-Ear Headphones",   # 二级品类
    "description": "...",                   # 合并后的描述文本
    "specifications": {                     # 结构化规格
        "Item Weight": "8.96 ounces",
        "Batteries": "1 Lithium Ion"
    },
    "rating": 4.6,                          # 平均评分
    "rating_count": 1234,                   # 评价数量
    "image_url": "https://..."              # 商品图片
}
```

---

## 5. 索引构建详解

### 5.1 双存储架构

`data/build_index.py` 同时构建两个索引：

| 存储 | 用途 | 内容 |
|---|---|---|
| **ChromaDB** | 语义检索 | 产品文本 embedding + 层级 metadata |
| **SQLite** | 确定性查询 | 完整结构化字段 + 聚合统计表 |

### 5.2 ChromaDB 索引

**Embedding 模型**: `all-MiniLM-L6-v2` (384 维，本地运行)  
**距离度量**: cosine  
**批量大小**: 256 条/批  

每条商品的 embedding 文本由以下字段拼接而成：

```
{title}. Model: {model}. Brand: {brand}.
Category: {main_category} > {subcategory}.
{description[:800]}
Specs: {key1: val1, key2: val2, ...}.
```

ChromaDB metadata 中存储了层级过滤所需的字段：

```python
{
    "title": "...",
    "brand": "Sony",           # Level-2 过滤键
    "model": "WH-1000XM4",
    "price": 348.0,            # 价格过滤
    "main_category": "...",    # Level-1 过滤键
    "subcategory": "...",
    "rating": 4.6,
    "rating_count": 1234,
    "has_price": 1             # 是否有标价
}
```

### 5.3 SQLite 表结构

**products 表**: 主商品表，12 个字段

```sql
CREATE TABLE products (
    id              TEXT PRIMARY KEY,
    title           TEXT,
    brand           TEXT,
    model           TEXT,
    price           REAL,
    main_category   TEXT,
    subcategory     TEXT,
    description     TEXT,
    specifications  TEXT,    -- JSON string
    rating          REAL,
    rating_count    INTEGER,
    image_url       TEXT
);
```

**brand_stats 表**: 品牌级聚合统计，供 `brand_summary` 工具使用

```sql
CREATE TABLE brand_stats (
    brand           TEXT,
    main_category   TEXT,
    product_count   INTEGER,
    avg_price       REAL,
    avg_rating      REAL,
    PRIMARY KEY (brand, main_category)
);
```

**category_tree 表**: 品类-品牌交叉统计，供层级浏览使用

```sql
CREATE TABLE category_tree (
    main_category   TEXT,
    subcategory     TEXT,
    brand           TEXT,
    product_count   INTEGER,
    PRIMARY KEY (main_category, subcategory, brand)
);
```

---

## 6. 三层级检索系统

### 6.1 架构总览

```
                    用户 query
                        │
            ┌───────────┴───────────┐
            │   Level 1 — 品类过滤   │  browse_categories()
            │   main_category       │  "Cell Phones & Accessories"
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │   Level 2 — 品牌过滤   │  browse_brands(category)
            │   brand               │  "Samsung", "Apple", ...
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │   Level 3 — 语义检索   │  retrieve_products(query, ...)
            │   cosine similarity   │  top-k 最相关商品
            └───────────┬───────────┘
                        │
                  候选商品列表
```

### 6.2 模块: `rag/retriever.py`

#### Level 1 — 品类浏览

```python
from rag.retriever import browse_categories

cats = browse_categories()
# 返回:
# [
#   {"category": "Cell Phones & Accessories", "product_count": 2000,
#    "brands": 967, "avg_rating": 4.03},
#   {"category": "All Electronics", "product_count": 2000, ...},
#   ...
# ]
```

#### Level 2 — 品牌浏览

```python
from rag.retriever import browse_brands

brands = browse_brands("Cell Phones & Accessories", min_products=3)
# 返回:
# [
#   {"brand": "Generic", "product_count": 100, "avg_price": 13.79, "avg_rating": 4.21},
#   {"brand": "OtterBox", "product_count": 61, "avg_price": 36.45, "avg_rating": 4.35},
#   {"brand": "SAMSUNG", "product_count": 29, ...},
#   ...
# ]
```

#### Level 2.5 — 子品类浏览

```python
from rag.retriever import browse_subcategories

subs = browse_subcategories(category="All Electronics")
# 返回:
# [
#   {"main_category": "All Electronics", "subcategory": "Earbud Headphones", "count": 249},
#   {"main_category": "All Electronics", "subcategory": "Wall Chargers", "count": 99},
#   ...
# ]
```

#### Level 3 — 语义检索

```python
from rag.retriever import retrieve_products

products = retrieve_products(
    query="wireless noise cancelling headphones for travel",
    category="All Electronics",       # Level-1 过滤 (可选)
    brand="Sony",                     # Level-2 过滤 (可选)
    top_k=10,
    min_price=50.0,
    max_price=500.0,
)
# 返回: list[dict]，每个 dict 包含:
#   id, title, brand, model, price, main_category,
#   subcategory, rating, rating_count, relevance_score
```

**关键设计**: 层级过滤条件被下推到 ChromaDB 的 `WHERE` 子句中，这意味着 ANN 搜索只在目标分段内进行，不是先检索再过滤，效率更高。

### 6.3 辅助函数

```python
from rag.retriever import get_product_detail, get_products_by_ids

detail = get_product_detail("B0BXYQJDJ5")    # 查单条完整记录
products = get_products_by_ids(["B0BX...", "B0BY..."])  # 批量查
```

---

## 7. 13个确定性工具

### 7.1 工具总览

所有工具定义在 `agent/tools.py` 中，并通过 `TOOL_REGISTRY` 字典暴露：

| # | 函数名 | 类型 | 输入 | 输出 |
|---|---|---|---|---|
| 1 | `apply_price_filter` | 过滤 | products + min/max_price | 过滤后的 products |
| 2 | `filter_by_brand` | 过滤 | products + allowed/excluded brands | 过滤后的 products |
| 3 | `filter_by_rating` | 过滤 | products + min_rating + min_count | 过滤后的 products |
| 4 | `filter_by_specs` | 过滤 | products + required_specs dict | 过滤后的 products (查 SQLite) |
| 5 | `spec_lookup` | 查询 | product_id | 完整规格 dict |
| 6 | `compare_products` | 分析 | product_ids list | side-by-side 对比表 |
| 7 | `rank_candidates` | 排序 | products + sort_by + limit | 排序后的 top-N |
| 8 | `find_similar_products` | 检索 | product_id + top_k | 相似商品列表 |
| 9 | `brand_summary` | 分析 | brand + category | 品牌统计汇总 |
| 10 | `price_statistics` | 分析 | category + brand | 价格分布 (min/max/avg/median/p25/p75) |
| 11 | `keyword_search` | 检索 | keyword + category + brand | SQL LIKE 匹配结果 |
| 12 | `product_detail` | 查询 | product_id | 完整商品记录 |
| 13 | `simulate_purchase` | 动作 | product_id | 模拟订单确认 |

### 7.2 各工具调用示例

#### Tool 1: 价格过滤

```python
from agent.tools import apply_price_filter

filtered = apply_price_filter(products, max_price=100.0, min_price=20.0)
```

#### Tool 4: 规格过滤

```python
from agent.tools import filter_by_specs

filtered = filter_by_specs(
    products,
    required_specs={"Screen Size": "6", "Batteries": "Lithium"}
)
```

该工具会查询 SQLite 获取完整规格数据，对每个 required key-value 做 **大小写不敏感的子串匹配**。

#### Tool 6: 商品对比

```python
from agent.tools import compare_products

table = compare_products(["B0BYK2THK8", "B0BYVRCLZV"])
# 返回:
# {
#   "columns": ["B0BYK2THK8", "B0BYVRCLZV"],
#   "rows": [
#     {"field": "title", "B0BYK2THK8": "...", "B0BYVRCLZV": "..."},
#     {"field": "brand", "B0BYK2THK8": "Funsnow", "B0BYVRCLZV": "EDYELL"},
#     {"field": "price", ...},
#     {"field": "spec:Item Weight", ...},
#     ...
#   ]
# }
```

前端可以直接用 `columns` 和 `rows` 渲染表格。

#### Tool 8: 相似商品推荐

```python
from agent.tools import find_similar_products

similar = find_similar_products("B0BXYQJDJ5", top_k=5, same_category=True)
```

原理: 取目标商品的标题+品牌+描述作为 query，在同品类内做语义检索，排除自身。

#### Tool 9: 品牌汇总

```python
from agent.tools import brand_summary

info = brand_summary("Samsung", category="Cell Phones & Accessories")
# 返回:
# {
#   "brand": "Samsung",
#   "stats": {"product_count": 29, "min_price": 8.99, "max_price": 999.99,
#             "avg_price": 206.56, "avg_rating": 4.03, "subcategories": "..."},
#   "top_rated": [{"id": "...", "title": "...", "rating": 4.9, "price": 199.0}, ...]
# }
```

#### Tool 10: 价格分布

```python
from agent.tools import price_statistics

stats = price_statistics("All Electronics")
# 返回:
# {"count": 1329, "min_price": 1.0, "max_price": 5197.57,
#  "avg_price": 76.83, "median_price": 19.89, "p25": 11.99, "p75": 39.99}
```

#### Tool 13: 模拟下单

```python
from agent.tools import simulate_purchase

receipt = simulate_purchase("B0BRNHL1HG")
# 返回:
# {
#   "success": true,
#   "order_id": "ORD-B0BRNHL1HG-20260412163749",
#   "product_id": "B0BRNHL1HG",
#   "product_title": "USB 3.0 Hub ...",
#   "price": 7.99,
#   "message": "Order placed for '...' at $7.99."
# }
```

### 7.3 TOOL_REGISTRY

所有工具通过一个 dict 注册，方便 Agent 编排层动态调用：

```python
from agent.tools import TOOL_REGISTRY

tool_fn = TOOL_REGISTRY["apply_price_filter"]
result = tool_fn(products, max_price=50.0)
```

---

## 8. RAG Pipeline 主管道

### 8.1 入口: `rag/pipeline.py`

这是其他角色对接本模块的 **唯一入口**，所有功能都通过这个文件暴露。

### 8.2 核心函数: `search()`

```python
from rag.pipeline import search

result = search(
    query="wireless headphones under $100 for running",
    constraints={
        "category": "All Electronics",
        "max_price": 100.0,
        "min_rating": 4.0,
        "sort_by": "rating",
        "limit": 5,
    },
)
```

#### 内部执行流程

```
1. 解析 constraints dict → SearchConstraints dataclass
       ↓
2. 调用 retriever.retrieve_products()
   ├── 构建 ChromaDB WHERE 过滤 (category + brand + price)
   └── 执行 ANN 语义检索，返回 top_k 候选
       ↓
3. 工具链式过滤
   ├── apply_price_filter()    (如果有 min/max_price)
   ├── filter_by_brand()       (如果有 allowed/excluded_brands)
   ├── filter_by_rating()      (如果有 min_rating/min_rating_count)
   └── filter_by_specs()       (如果有 required_specs)
       ↓
4. rank_candidates()           (按 sort_by 排序，取 top limit)
       ↓
5. 返回 PipelineResult
```

#### SearchConstraints 完整参数

```python
@dataclass
class SearchConstraints:
    category: str | None          # Level-1 品类过滤
    subcategory: str | None       # 子品类过滤
    brand: str | None             # Level-2 品牌过滤 (ChromaDB 下推)
    max_price: float | None       # 价格上限
    min_price: float | None       # 价格下限
    allowed_brands: list[str]     # 品牌白名单 (工具层过滤)
    excluded_brands: list[str]    # 品牌黑名单
    min_rating: float | None      # 最低评分
    min_rating_count: int         # 最少评价数
    required_specs: dict[str,str] # 规格要求
    sort_by: str                  # 排序字段: relevance_score|price|rating|rating_count
    top_k: int                    # 检索候选数量 (默认 10)
    limit: int                    # 最终返回数量 (默认 5)
```

#### PipelineResult 输出结构

```python
@dataclass
class PipelineResult:
    query: str                    # 输入的查询
    constraints_applied: dict     # 实际应用的约束摘要
    retrieved_count: int          # 语义检索命中数
    filtered_count: int           # 工具过滤后剩余数
    products: list[dict]          # 最终推荐的商品列表
    comparison: dict | None       # (可选) 对比表
    checkout: dict | None         # (可选) 下单结果
```

`PipelineResult` 提供两个序列化方法:

```python
result.to_dict()   # → dict
result.to_json()   # → JSON string
```

### 8.3 便捷函数清单

`rag/pipeline.py` 还暴露了以下便捷函数，供 Agent 直接调用：

| 函数 | 说明 | 对应工具 |
|---|---|---|
| `explore_categories()` | 浏览所有品类 | retriever.browse_categories |
| `explore_brands(category)` | 浏览品牌 | retriever.browse_brands |
| `explore_subcategories(category, brand)` | 浏览子品类 | retriever.browse_subcategories |
| `compare(product_ids)` | 商品对比 | tools.compare_products |
| `checkout(product_id)` | 模拟下单 | tools.simulate_purchase |
| `similar(product_id)` | 相似推荐 | tools.find_similar_products |
| `detail(product_id)` | 查完整信息 | tools.product_detail |
| `specs(product_id)` | 查规格 | tools.spec_lookup |
| `brand_info(brand, category)` | 品牌汇总 | tools.brand_summary |
| `price_stats(category, brand)` | 价格分布 | tools.price_statistics |
| `text_search(keyword, ...)` | 关键词兜底搜索 | tools.keyword_search |

---

## 9. 集成指南 — 如何对接 Agent 编排层

### 9.1 基本搜索

```python
from rag.pipeline import search

# 对话 Agent 完成意图解析后，传入改写 query + 结构化约束
result = search(
    query="iPhone 14 protective case military grade drop protection",
    constraints={
        "category": "Cell Phones & Accessories",
        "max_price": 30.0,
        "min_rating": 4.0,
        "limit": 5,
    },
)

# 把结果喂给 LLM 生成推荐文本
for p in result.products:
    print(f"{p['title']} — ${p['price']} — Rating {p['rating']}")
```

### 9.2 层级钻取 (对话中逐步缩小范围)

```python
from rag.pipeline import explore_categories, explore_brands, search

# 第 1 轮: 用户说 "我想买电子产品"
cats = explore_categories()
# → Agent 向用户展示品类列表，让用户选择

# 第 2 轮: 用户说 "手机配件"
brands = explore_brands("Cell Phones & Accessories", min_products=5)
# → Agent 展示热门品牌，让用户选择或直接搜索

# 第 3 轮: 用户说 "OtterBox 的手机壳，50块以下"
result = search("phone case", constraints={
    "category": "Cell Phones & Accessories",
    "brand": "OtterBox",
    "max_price": 50.0,
})
```

### 9.3 LangChain Tool 注册示例

Agent 编排层 (角色 2) 可以这样注册 Tool:

```python
from langchain.tools import tool
from rag.pipeline import search, compare, checkout, explore_categories, explore_brands

@tool
def search_products(query: str, category: str = None, brand: str = None,
                    max_price: float = None, min_rating: float = None) -> str:
    """Search for consumer electronics products with optional filters."""
    result = search(query, constraints={
        "category": category, "brand": brand,
        "max_price": max_price, "min_rating": min_rating,
    })
    return result.to_json()

@tool
def compare_products_tool(product_ids: list[str]) -> str:
    """Compare 2-4 products side by side."""
    return json.dumps(compare(product_ids))

@tool
def browse_categories_tool() -> str:
    """List all available product categories."""
    return json.dumps(explore_categories())

@tool
def browse_brands_tool(category: str) -> str:
    """List popular brands in a category."""
    return json.dumps(explore_brands(category))

@tool
def checkout_tool(product_id: str) -> str:
    """Simulate purchasing a product."""
    return json.dumps(checkout(product_id))
```

---

## 10. 集成指南 — 如何对接前端

### 10.1 搜索结果 → 商品卡片

```python
result = search("wireless earbuds")

# result.products 中每个 dict 可直接用于渲染卡片:
for p in result.products:
    st.card(
        title=p["title"],
        subtitle=f"{p['brand']} · ${p['price']:.2f}" if p['price'] else p['brand'],
        metric=f"Rating: {p['rating']} ⭐",
        badge=p["main_category"],
    )
```

### 10.2 对比表 → Streamlit 表格

```python
comp = compare(["B0BYK2THK8", "B0BYVRCLZV"])
import pandas as pd

df = pd.DataFrame(comp["rows"])
df = df.set_index("field")
st.dataframe(df)
```

### 10.3 品类/品牌浏览 → 侧边栏

```python
cats = explore_categories()
selected_cat = st.sidebar.selectbox(
    "Category",
    [c["category"] for c in cats],
)

brands = explore_brands(selected_cat)
selected_brand = st.sidebar.selectbox(
    "Brand",
    ["All"] + [b["brand"] for b in brands],
)
```

---

## 11. 测试结果

### 11.1 测试概况

运行命令: `python -X utf8 -m scripts.test_pipeline`  
结果: **15/15 测试全部通过**

### 11.2 各测试用例详情

| # | 测试名 | 目标 | 结果 | 关键数据 |
|---|---|---|---|---|
| 1 | Browse categories | Level-1 品类浏览 | ✅ | 5 个品类，最大 2000 条 |
| 2 | Browse brands | Level-2 品牌浏览 | ✅ | Cell Phones 下 10+ 品牌 |
| 3 | Browse subcategories | 子品类浏览 | ✅ | All Electronics 下 Earbud Headphones 最多 (249) |
| 4 | Basic semantic search | 无约束语义搜索 | ✅ | 检索 10 条，top-1 相关度 0.68 |
| 5 | Category-filtered search | 品类限定搜索 | ✅ | 只返回 Cell Phones 品类 |
| 6 | Brand-filtered search | 品牌限定搜索 | ✅ | 只返回 Anker 品牌 (4 条) |
| 7 | Price-filtered search | 价格约束搜索 | ✅ | 所有结果 ≤ $30 |
| 8 | Combined constraints | 多约束组合搜索 | ✅ | Computers + $50-200 + rating≥4.0，按评分排序 |
| 9 | Compare products | 商品对比表 | ✅ | 2 商品 side-by-side，含基础字段和规格 |
| 10 | Mock checkout | 模拟下单 | ✅ | 成功生成订单号和确认信息 |
| 11 | Similar products | 相似推荐 | ✅ | 为一款儿童平板推荐了 3 款同类产品 |
| 12 | Brand summary | 品牌汇总 | ✅ | 返回品牌级统计信息 |
| 13 | Price statistics | 价格分布 | ✅ | All Electronics: 中位价 $19.89，P75 $39.99 |
| 14 | Keyword search | 关键词兜底 | ✅ | "charger" 在 Cell Phones 中命中 5 条 |
| 15 | Spec lookup | 规格查询 | ✅ | Camera 品类商品返回尺寸、重量等规格 |

### 11.3 检索质量示例

**查询**: "wireless bluetooth headphones noise cancelling"  
**结果** (按相关度排序):

| 排名 | 商品 | 品牌 | 价格 | 评分 | 相关度 |
|---|---|---|---|---|---|
| 1 | E7 Active Noise Cancelling Headphones | Generic | $29.98 | 4.6 | 0.679 |
| 2 | Bluetooth Wireless Headphones | Generic | N/A | 5.0 | 0.669 |
| 3 | Wireless Earbud Bluetooth 5.3 | Jesebang | N/A | 4.9 | 0.632 |
| 4 | Sony WH-1000XM4 | Sony | $348.00 | 1.0 | 0.620 |
| 5 | Wireless Earbuds Bluetooth 5.0 | Moxil | N/A | 4.3 | 0.613 |

语义检索能正确把降噪耳机排在前列，且涵盖不同价位和品牌。

---

## 12. 性能与成本分析

### 12.1 构建时间

| 步骤 | 耗时 | 说明 |
|---|---|---|
| 数据下载+清洗 | ~90 秒 | 首次需要网络，后续使用缓存 |
| ChromaDB 索引 | ~4 分钟 | 6784 条 × all-MiniLM-L6-v2 embedding |
| SQLite 建表 | < 1 秒 | 纯 INSERT |

### 12.2 查询时间

| 操作 | 耗时 |
|---|---|
| 首次加载 embedding 模型 | ~2 秒 (一次性) |
| 语义检索 (单次 query) | ~50ms |
| 工具链过滤 (4 个工具) | < 10ms |
| 端到端 search() | ~60ms |

### 12.3 存储占用

| 文件 | 大小 |
|---|---|
| `products_clean.json` | ~25 MB |
| `chroma_db/` | ~55 MB |
| `products.db` | ~15 MB |

### 12.4 LLM Token 成本

**本模块的 token 消耗为零**。所有检索和工具调用都在本地完成：
- Embedding 使用本地 `all-MiniLM-L6-v2` 模型
- 所有过滤、排序、统计使用 Python + SQLite
- 只有最终推荐文本生成由上游 Agent 的 LLM 负责

---

## 13. 已知限制与后续优化

### 13.1 当前限制

| 项目 | 说明 |
|---|---|
| **价格缺失** | 数据集中约 60% 的商品没有标价，价格过滤会保留无价商品 |
| **品牌噪音** | 部分商品 brand 为 "Generic" 或 "Unknown"，占约 6% |
| **规格稀疏** | 并非所有商品都有结构化规格字段 |
| **评论文本** | 当前只使用了评分和评价数量，没有利用评论原文 |
| **实时性** | 数据来自 2023 年快照，不含最新产品 |

### 13.2 后续优化方向

1. **替换为更大的数据集**: 当 Kaggle 上有更完整的数据集时，只需修改 `load_dataset.py` 的数据源
2. **多向量策略**: 为标题、描述、评论分别建向量，用 late fusion 合并相关度
3. **评论情感分析**: 引入评论原文做情感摘要，作为推荐理由
4. **重排序 (Re-ranking)**: 在语义检索后加一个 cross-encoder 重排，提升精度
5. **缓存层**: 对高频查询缓存结果，减少重复计算

---

## 附录 A: 完整 API 速查表

```python
# ── 主管道 ──
from rag.pipeline import search, SearchConstraints, PipelineResult

# ── 层级浏览 ──
from rag.pipeline import explore_categories, explore_brands, explore_subcategories

# ── 工具调用 ──
from rag.pipeline import compare, checkout, similar, detail, specs
from rag.pipeline import brand_info, price_stats, text_search

# ── 底层接口 (一般不需要直接调用) ──
from rag.retriever import retrieve_products, get_product_detail
from agent.tools import TOOL_REGISTRY
```

## 附录 B: 快速验证命令

```bash
# 验证安装
python -c "import chromadb; from sentence_transformers import SentenceTransformer; print('OK')"

# 验证索引
python -c "from rag.pipeline import explore_categories; print(explore_categories())"

# 验证搜索
python -c "from rag.pipeline import search; r=search('headphones'); print(r.to_json())"

# 完整测试
python -X utf8 -m scripts.test_pipeline
```
