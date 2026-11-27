from typing import Any

from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger


# 步骤1: 参数校验和获取
def validate_and_get_chunks(state: ImportGraphState) -> list[dict[str, Any]]:
    # 1.1 在state中获取chunks数据
    chunks: list[dict[str, Any]] = state.get("chunks")
    # 1.2 进行chunks非空校验
    if not chunks:
        logger.error(f"chunks内容为空，业务无法继续，提前终止！")
        raise ValueError(f"chunks内容为空，业务无法继续，提前终止！")
    # 返回chunks
    return chunks


# 步骤2: 批量chunks向量化处理
def batch_chunks_embedding(chunks: list[dict[str, Any]]):
    # 2.1 获取chunks内容长度 len()
    total = len(chunks)
    # 2.2 指定步长开始进行处理chunks  for index in range(0,len,step):
    for index in range(0, total, EMBEDDING_BATCH_SIZE):
        logger.debug(f"进行{index}~{index + EMBEDDING_BATCH_SIZE}批次的chunks向量化处理!")
        # 2.3 在循环中截取本次批量处理的sub_chunks =chunks[index:index+step]
        sub_chunks: list[dict[str, Any]] = chunks[index:index + EMBEDDING_BATCH_SIZE]
        # 2.4 sub_chunks每个chunk的item_name+content进行处理拼接 list[str]
        item_name_and_content: list[str] = []
        for sub_chunk in sub_chunks:
            item_name_and_content.append(
                # content -> 稠密和稀疏 <- 问题 : 他好吃么? 重写 [沙琪玛 item_name]好吃么?
                sub_chunk['item_name'] + ":" + sub_chunk['content']
            )
        # 2.5 使用模型的infra批量为list[str]生成向量,向量结果dict[str,list]
        result: dict[str, list] = llm_provider.generate_infra_embeddings(item_name_and_content)
        # 2.6 进行数据回填 向量 -> sub_chunks中即可 [顺序是对应]
        for sub_index in range(len(sub_chunks)):
            # sub_index  0  1 2 3 4
            dense_vector = result['dense'][sub_index]
            sparse_vector = result['sparse'][sub_index]
            # sub_chunks[chunk:dict[str,Any]  {title:x,content:xx,dense_vector: , sparse_vector: },chunk,chunk]
            sub_chunks[sub_index]['dense_vector'] = dense_vector
            sub_chunks[sub_index]['sparse_vector'] = sparse_vector
    # 2.7 跳出循环打印日志
    logger.info(f"已经完成所有切片的内容向量化了! 打印示例: {chunks[0]}")


def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    # 参数校验和获取
    chunks: list[dict[str, Any]] = validate_and_get_chunks(state)
    # 批量向量化处理
    batch_chunks_embedding(chunks)
    # 更新state
    state["embeddings_content"] = chunks
    # 返回state
    return state
