from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.entry_service import resolve_input_file

# 装饰器的作用: 打印日志
@node_log("node_entry")
# 节点函数 ( state 全局共享域 |  runtime context)
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """

    # add_running_task  add_done_task 函数 -> 来自于我们自定义的工具类  shared / utils / task_utils.py [大致了解!web接口的时候再详细说]
    # 简单的理解: 将进行中或者完成的节点存储到容器中 -> 容器 <- fastapi节点读取 <- 轮询 1s <- 前端页面
    # 为啥不是单独添加节点名,还添加了一个task_id?  底层存储使用字典,为每个客户端存储一个对应的列表! task_id客户端的标识!!
    #  {task_id: str}  -> {task_id:list}
    add_running_task(state["task_id"], "node_entry")
    # langgraph -> 节点函数 -> 业务函数
    # 参数: state  业务操作 数据操作  返回: state
    state = resolve_input_file(state)
    add_done_task(state["task_id"], "node_entry")

    # state数据返回  langgraph进行合并传递到下一个节点
    return state