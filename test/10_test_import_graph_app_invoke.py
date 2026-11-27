from pathlib import Path

from app.process.import_.agent.main_graph import import_graph_app
from app.process.import_.agent.state import create_default_state,ImportGraphState
from app.shared.runtime.logger import PROJECT_ROOT,logger
import uuid
import json

# 1. 创建state [第一个节点需要的内容]
# (local_file_path = 要处理的文件地址  local_dir = 输出文件的文件夹  task_id = 任务的唯一标识 )
local_file_path_obj:Path = PROJECT_ROOT / "doc" / "华为擎云W585 用户指南-(PGUV,KOS&UOS_01,zh-cn).pdf"
local_dir_obj:Path = PROJECT_ROOT / "output"
task_id:str = str(uuid.uuid4())

state:ImportGraphState = create_default_state(
    task_id = task_id,
    local_file_path = str(local_file_path_obj),
    local_dir = str(local_dir_obj)
)

logger.info(f"执行参数:\n{json.dumps(state,indent=4,ensure_ascii=False)}")

# 2. 执行langgraph_app图对象获取结果
result:ImportGraphState= import_graph_app.invoke(state)

# 3. 输出结果即可
logger.info(f"最终结果:\n{json.dumps(result,indent=4,ensure_ascii=False)}")

