import inspect
import os
from enum import Enum

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

ALI_TONGYI_API_KEY_SYSVAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
ALI_TONGYI_EMBEDDING = "text-embedding-v3"
ALI_TONGYI_RERANK = "gte-rerank-v2"
DEEPSEEK_API_KEY_OS_VAR_NAME = "Deepseek_Key"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"


class Constants(Enum):
    API_KEY_SYSVAR_NAME = ALI_TONGYI_API_KEY_SYSVAR_NAME
    BASE_URL = ALI_TONGYI_URL
    LLM_MODEL = ALI_TONGYI_MAX_MODEL
    EMBEDDING_MODEL = ALI_TONGYI_EMBEDDING
    RERANK_MODEL = ALI_TONGYI_RERANK
    REASONER_MODEL = ALI_TONGYI_DEEPSEEK_R1


def get_lc_model_client(
        api_key=os.getenv(Constants.API_KEY_SYSVAR_NAME.value),
        base_url=Constants.BASE_URL.value,
        model=Constants.LLM_MODEL.value,
        verbose=False,
        temperature=0.7,
        debug=False):
    function_name = inspect.currentframe().f_code.co_name
    if (verbose):
        print(f"{function_name}:{base_url},{model}")
    if (debug):
        print(f"{function_name}:{base_url},{model},{api_key}")
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model)


def get_ali_model_client(
        model=ALI_TONGYI_MAX_MODEL,
        temperature=0.7,
        verbose=False,
        debug=False):
    '''
    通过Langchain获得阿里大模型的客户端
    可以通过传入model，temperature 两个参数来覆盖默认值
    :param model:
    :param temperature:
    :param verbose: 控制是否输出调试信息，默认False
    :param debug: 是否输出详细的调试信息，默认False
    :return:
    '''
    return get_lc_model_client(
        api_key=os.getenv(Constants.API_KEY_SYSVAR_NAME.value),
        base_url=ALI_TONGYI_URL,
        model=model,
        temperature=temperature,
        verbose=verbose,
        debug=debug)


def get_ali_embeddings():
    '''
    通过Langchain来获得一个阿里通义千问嵌入模型实例
    :return: 阿里通义千问嵌入模型实例，目前为text-embedding-v3
    '''
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING,
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_SYSVAR_NAME)
    )


def get_ali_clients():
    '''
    产生阿里大模型客户端和嵌入模型的客户端
    :return: 阿里大模型客户端 和 嵌入模型客户端
    '''
    return get_ali_model_client(), get_ali_embeddings()
