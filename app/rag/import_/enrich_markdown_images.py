import base64
import re
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio import Minio
from minio.deleteobjects import DeleteObject

from app.infra.llm.providers import llm_provider
from app.infra.object_storage.minio_gateway import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS, IMAGE_CONTEXT_CHAR_LENGTH
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger

"""
1. 获取核心参数和数据校验
目标: state 获取: md_path_obj images_path_obj md_content
validates_and_get_data(state:ImportGraphState) -> tuple[str,Path,Path]
1.1 获取md_path参数从state
1.2 md_path的非空校验
1.3 转成md_path_obj Path对象,做存在性校验! is_file()
1.4 使用md_path对象读取md_content,更新state
1.5 md_path_obj可以获取images path地址
1.6 顺序返回 return md_content,..
"""


def validates_and_get_data(state: ImportGraphState) -> tuple[str, Path, Path]:
    md_path: str = state.get("md_path")

    if not md_path:
        logger.error(f"md_path内容为空,业务无法继续,提前终止!")
        raise ValueError(f"md_path内容为空,业务无法继续,提前终止!")

    md_path_obj: Path = Path(md_path)

    if not md_path_obj.is_file():
        logger.error(f"md_path地址有值:{md_path},但是没有对应的文件,业务无法继续,提前终止!")
        raise FileNotFoundError(f"md_path地址有值:{md_path},但是没有对应的文件,业务无法继续,提前终止!")

    md_content: str = md_path_obj.read_text(encoding="utf-8")
    state["md_content"] = md_content

    # 1.5 md_path_obj可以获取images path地址
    images_path_obj: Path = md_path_obj.parent / "images"

    return md_content, md_path_obj, images_path_obj


"""
scan_images_info(images_path_obj:Path,md_content:str) -> list[tuple[str,Path,tuple[str,str]]]
        2.1 获取images下的所有图片Path对象 [列表/生成器]
        2.2 循环每一张图片对象
            2.2.1 定义这个图片的正则规则
            2.2.2 使用正则到md_content中进行匹配
            2.2.3 检查是否匹配 Match是否为空
            2.2.4 获取对应上下信息
            2.2.5 组装当前的数据元组 (图片名字str,图片Path对象,(上文,下文)) 添加到大的列表
        2.3 返回列表数据

"""


def scan_images_info(images_path_obj: Path, md_content: str) -> list[tuple[str, Path, tuple[str, str]]]:
    # 2.1 获取images下的所有图片Path对象 [列表/生成器]
    # images_path_obj   iterdir() | glob("*.jpg")/rglob("规则")
    generator = images_path_obj.iterdir()
    images_info_list: list[tuple[str, Path, tuple[str, str]]] = []
    # 2.2 循环每一张图片对象
    for image_file_obj in generator:
        # 检查是不是图片
        if image_file_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.debug(f"{image_file_obj.name}不是图片，跳过，进行下一张处理！")
            continue
        # 2.3 定义这个图片的正则规则
        image_name: str = image_file_obj.name
        match = re.search(r"\!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)", md_content)
        # 2.2.3 检查是否匹配 Match是否为空
        if not match:
            logger.debug(f"{image_name}没有在md_content被使用,跳过,进行下一张处理!")
            continue
        # 获取对应上下文信息
        start = match.start()
        end = match.end()
        # xxx ! []( ) xxxx
        # IMAGE_CONTEXT_CHAR_LENGTH
        # [)
        # str[,start ! 不要] -> 字符串截取
        # [1 负值 0 最小值 ,2 没问题]
        pre_context: str = md_content[max(start - IMAGE_CONTEXT_CHAR_LENGTH, 0): start]
        post_context: str = md_content[end + 1: min(end + IMAGE_CONTEXT_CHAR_LENGTH, len(md_content))]
        images_info_list.append(
            (
                image_name,
                image_file_obj,
                (
                    pre_context,
                    post_context,
                )
            )
        )
    logger.info(f"已经完成了{len(images_info_list)}张图片的匹配")
    return images_info_list


"""
summary_image_info(image_info:list[tuple[str,Path,tuple[str,str]]],root_folfer:str) -> dict[str,str]
    3.1 数据准备工作 初始化一个视觉模型对象 | 定义一个接收数据的字典
    3.2 循环处理每一张图片    for str,Path,context in  list[tuple[str,Path,tuple[str,str]]]
        3.3 封装模型提示词信息 
        3.4 封装调用链: 直接 model.invoke | 调用链 提示词 | 模型 | 解析器 OutParser() 
        3.5 使用调用链进行调用和获取结果
        3.6 添加到字典中 {image_name,image_summary}
    3.7 跳出循环->返回整体字典
"""


def summary_image_info(image_info: list[tuple[str, Path, tuple[str, str]]], root_folfer: str) -> dict[str, str]:
    # 3.1 数据准备工作 初始化一个视觉模型对象 | 定义一个接收数据的字典
    images_summary: dict[str, str] = {}
    vision_model = llm_provider.vision_chat()
    # 3.2 环处理每一张图片    for str,Path,context in  list[tuple[str,Path,tuple[str,str]]]
    for image_name, image_path_obj, image_context in image_info:
        # 3.3 封装模型提示词信息
        # 3.3.1 图片的描述 包裹上下文信息一段提示词
        image_des: str = load_prompt("image_summary", image_content=image_context, root_folder=root_folfer)
        image_type: str = guess_type(image_name)[0]  # (类型,None)
        image_base64: str = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": image_des},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_type};base64,{image_base64}"},
                },
            ]
        )
        # 3.4 封装调用链: 直接 model.invoke | 调用链 提示词 | 模型 | 解析器 OutParser()
        # chains = 提示词 | 模型 | 解析器
        chains = vision_model | StrOutputParser()
        # 3.5 使用调用链进行调用和获取结果
        summary: str = chains.invoke([message])
        # model.invoke([]) -> response -> response . content -> 字符串
        # chains = vision_model | JSONOutputParser()
        # model.invoke([]) -> response -> response . content -> 字符串 -> json.loads(字符串) -> dict
        images_summary[image_name] = summary
        logger.debug(f"图片:{image_name},已经通过视觉模型进行意图识别,结果为:{summary}")
    logger.info(f"已经完成{root_folfer}文件夹中的所有图片内容总结,总结图的内容对数为:{len(images_summary)}")
    return images_summary


"""
upload_image_to_minio(image_info:list[tuple[str,Path,tuple[str,str]]],stem) -> dict[str,str]:
        4.1 前置准备工作 获取minio_client / 创建一个存储结果的dict
        4.2 根据桶信息查询对应旧图片对象
        4.3 批量删除旧图片对象
        4.4 循环处理每一张图片  for image_name,image_path_obj,_ in image_info:
            4.5 单次循环的异常处理,失败,跳过图片即可,进行下一张!
            4.6 调用minio_client.fput方法将图片上传到minio服务器
            4.7 根据信息调用infra构建图片的访问地址
            4.8 将图片访问地址存储到dict image_name | image_url
        4.9 最终返回dict数据
"""


def upload_image_to_minio(image_info: list[tuple[str, Path, tuple[str, str]]], stem) -> dict[str, str]:
    # 4.1 前置准备工作 获取minio_client / 创建一个存储结果的dict
    images_url: dict[str, str] = {}
    minio_client: Minio = minio_gateway.client
    # 4.2 根据桶信息查询对应旧图片对象
    objects: list = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        #  # 桶名  -> /图片前缀/文件名/  <- 图片.png
        # todo 注意: 要求必须不能使用 / 开始! -> [1:] 可以使用 / 结束或者 不添加 / 结束
        prefix=minio_gateway.image_dir[1:] + "/" + stem,
        recursive=True
    )
    # 4.3 批量删除旧图片对象
    if objects:
        # 查询到了之前文件有对应图片,进行删除
        """
              Object 查询对象  DeleteObject 删除对象  <- minio 
          坑1: list_objects -> list ->  list[Object]  || list[DeleteObject] <- delete_object_list <- remove_objects 
                     list[Object] -> object_name -> list[DeleteObject]
          坑2: remove_objects方法返回的是一个生成器,需要循环处理结果,才会真的触发!!! 
               errors = x
               for error in errors:
                   pass
        """
        delete_object_list = [DeleteObject(obj.object_name) for obj in objects]
        errors = minio_client.remove_objects(
            bucket_name=minio_gateway.bucket_name,
            delete_object_list=delete_object_list
        )
        for error in errors:
            logger.warning(f"删除{stem}文件夹下的图片出现问题,问题:{error}")
    # 4.4 循环处理每一张图片  for image_name,image_path_obj,_ in image_info:
    for image_name, image_path_obj, _ in image_info:
        # 4.5 单次循环的异常处理,失败,跳过图片即可,进行下一张!
        try:
            # 4.6 调用minio_client.fput方法将图片上传到minio服务器
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=minio_gateway.image_dir + "/" + stem + "/" + image_name,
                # object_name -> 决定我们在桶中的文件结构!  桶 [不需要]  / 图片的前缀 / 文件名 / 图片名
                file_path=str(image_path_obj),  # Path -> str
                content_type=guess_type(image_name)[0]
            )
            # 不报错,成功
            # 4.7 根据信息调用infra构建图片的访问地址
            url: str = minio_gateway.build_image_url(stem, image_name)
            # 4.8 将图片访问地址存储到dict image_name | image_url
            images_url[image_name] = url
            logger.debug(f"{image_name}已经传递到了minio文件服务器,访问地址为:{url}")
        except Exception as e:
            logger.exception(f"{image_name}图书上传失败,跳过本次,进行下一张图片处理!")
            continue
    # 4.9 最终返回dict数据
    logger.info(f"已经{stem}所有图片的上传,上传的数量为:{len(images_url)}")
    return images_url


# 5. 完成md_content内部图片标签的替换,获得新的md_content内容
def replace_old_md_content(md_content: str, images_url: dict[str, str], images_summary: dict[str, str]) -> str:
    # 5.1  循环images_url可以获取每张图片的image_name / 网络地址
    for image_name, url in images_url.items():
        # 5.2 可以根据image_name去images_summary获取对应的summary描述
        summary = images_summary.get(image_name)
        # 5.3 定义提前编译查询原有图片标签的正则 re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+".*?\)")
        image_re = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)")
        # 5.4 使用正则的替换函数即可    替换后的内容 md_content  =  编译的对象.sub("替换入的内容" f"![summary](url)" ,"什么中查找替换" -> md_content)
        # 累加替换 1 2 3 4
        # f"![{summary}]({url})" 为啥没有添加 \? 因为这个地方就是一个普通的字符串,不是正则!!!
        # 天然一个问题: sub(" \ -> 反  / -> 正斜杠  替换中代表分组!!匹配错误!!")
        # 解决分组替换: lambda表达式,匿名函数
        # sub( 1 (替换入的字符串 -> 会检查 / 分组替换的特殊符号 | 匿名函数,匿名函数的返回结果替换入 -> 不是检查返回结果 ) , )
        # 记住: 以后所有的替换 第一个参数都不要直接写字符串  lambda _: 替换入的字符串
        md_content: str = image_re.sub(lambda _: f"![{summary}]({url})", md_content)
    return md_content


# 6. 将新md_content写到本地磁盘,持久化
def backup_md_return_path(md_content: str, md_path_obj: Path) -> str:
    # 6.1 获取新备份位置的Path 文件名_new.md  md_path_obj.with_name | rename
    new_md_path_obj: Path = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    # 6.2 向新的path对象写入新的md_content md_path_obj.write_text
    new_md_path_obj.write_text(data=md_content, encoding="utf-8")
    # 6.3 新的path对象转成字符串返回即可
    return str(new_md_path_obj)


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    # 参数校验
    md_content, md_path_obj, images_path_obj = validates_and_get_data(state)
    # md就没有图片,需要特殊的校验和提前终止  images_path_obj 文件夹 / iterdir 所有的文件或者子文件夹 |  glob rglob (*.后缀名) -> 生成器
    if not list(images_path_obj.iterdir()):
        logger.info(f"{md_path_obj}中没有引用任何图片，无需后续图片处理，直接跳转到下一个节点!!")
        return state
    # 获取图片信息
    images_info_list: list[tuple[str, Path, tuple[str, str]]] = scan_images_info(images_path_obj, md_content)
    # 调用模型进行图片内容总结
    images_summary: dict[str, str] = summary_image_info(images_info_list, md_path_obj.stem)
    # 调用图片上传，获取图片上传地址
    images_url: dict[str, str] = upload_image_to_minio(images_info_list, md_path_obj.stem)
    # 进行md_content内容替换
    new_md_content: str = replace_old_md_content(md_content, images_url, images_summary)
    # new_md_content数据备份
    new_md_path: str = backup_md_return_path(new_md_content, md_path_obj)
    # 更新state
    state["md_path"] = new_md_path
    state["md_content"] = new_md_content
    # 返回结束
    return state
