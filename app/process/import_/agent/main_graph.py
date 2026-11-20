from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus
from app.shared.runtime.logger import logger

# 创建图的构建对象
import_builder = StateGraph(state_schema=ImportGraphState)

# 添加节点
import_builder.add_node("node_entry", node_entry)
import_builder.add_node("node_pdf_to_md", node_pdf_to_md)
import_builder.add_node("node_md_img", node_md_img)
import_builder.add_node("node_document_split", node_document_split)
import_builder.add_node("node_item_name_recognition", node_item_name_recognition)
import_builder.add_node("node_bge_embedding", node_bge_embedding)
import_builder.add_node("node_import_milvus", node_import_milvus)

# 添加边
import_builder.set_entry_point("node_entry")


def after_entry_node(state: ImportGraphState):
    if state.get("is_md_read_enabled"):
        logger.info(f"传入文件是md格式，md_path:{state.get('md_path')},后续跳转到node_md_img节点")
        return "node_md_img"
    elif state.get("is_pdf_read_enabled"):
        logger.info(f"传入文件是pdf格式，pdf_path:{state.get('pdf_path')},后续跳转到后续跳转到node_md_img节点节点")
        return "node_pdf_to_md"
    else:
        logger.warning(f"当前项目只支持pdf和md格式，local_file_path:{state.get('local_file_path')},后续跳转到END节点！")
        return END

import_builder.add_conditional_edges(
    "node_entry",
    after_entry_node,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END,
    },
)

import_builder.add_edge("node_pdf_to_md", "node_md_img")
import_builder.add_edge("node_md_img", "node_document_split")
import_builder.add_edge("node_document_split", "node_item_name_recognition")
import_builder.add_edge("node_item_name_recognition", "node_bge_embedding")
import_builder.add_edge("node_bge_embedding", "node_import_milvus")
import_builder.add_edge("node_import_milvus", END)

# 编译生成图对象
import_graph_app = import_builder.compile()
