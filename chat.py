__all__ = ['chat']

import glob
import os
from functools import cache

from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
    ConversationalRetrievalChain
)
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate
)
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from embedding import embed_documents

_DEFAULT_PROMPT_TEMPLATE_DIR = os.path.abspath('res/templates/')
_PROMPT_TEMPLATE_SUFFIX = '-prompt-template.txt'


def _load_prompt_template(path_to_template: str) -> str:
    with open(path_to_template, 'r') as f:
        return ''.join(f.readlines())


def _load_available_prompt_templates(path_to_dir: str) -> dict[str, str]:
    path_to_dir = os.path.abspath(path_to_dir) + os.path.sep
    available_files: list[str] = glob.glob(f'{path_to_dir}*{_PROMPT_TEMPLATE_SUFFIX}')

    available_templates = {
        os.path.basename(file).removesuffix(_PROMPT_TEMPLATE_SUFFIX).lower():
            _load_prompt_template(file)
        for file in available_files
    }

    return available_templates


def _load_prompt_templates_from_dir(path_to_dir: str,
                                    *args: str,
                                    consolidate_with_defaults: bool = True) -> dict[str, str] | tuple[str | None, ...]:
    templates = _load_available_prompt_templates(path_to_dir) \
        if path_to_dir else _load_default_prompt_templates()

    if consolidate_with_defaults:
        templates = _load_default_prompt_templates() | templates

    if not args:
        return templates

    return tuple(templates[key] for key in args)


@cache
def _load_default_prompt_templates():
    return _load_prompt_templates_from_dir(_DEFAULT_PROMPT_TEMPLATE_DIR, consolidate_with_defaults=False)


def _create_model() -> BaseChatModel:
    return ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')


def _create_conversational_chain(vectorstore: VectorStore,
                                 model: BaseChatModel,
                                 prompt_template_dir: str,
                                 num_docs: int,
                                 verbose: bool) -> BaseConversationalRetrievalChain:
    template_keys = 'system-message', 'human-message', 'condense-question'
    (
        system_message_prompt_template,
        human_message_prompt_template,
        condense_question_prompt_template
    ) = _load_prompt_templates_from_dir(prompt_template_dir, *template_keys)

    system_message_prompt = SystemMessagePromptTemplate.from_template(template=system_message_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(template=human_message_prompt_template)
    question_answering_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    condense_question_prompt = PromptTemplate.from_template(template=condense_question_prompt_template)

    memory: ConversationBufferMemory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(search_kwargs={'k': num_docs}),
        condense_question_prompt=condense_question_prompt,
        verbose=verbose,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': question_answering_prompt}
    )


def _load_conversational_chain(docs_dirs: tuple[str],
                               embeddings_model: str,
                               prompt_template_dir: str,
                               num_docs: int,
                               overwrite_cached_embeddings: bool,
                               run_psql_instance: bool,
                               verbose: bool) -> BaseConversationalRetrievalChain:
    vectorstore: VectorStore = embed_documents(
        docs_dirs,
        (embeddings_model,),
        overwrite_cached_embeddings,
        run_psql_instance,
        exit_if_unknown_model=True
    )[0]

    model: BaseChatModel = _create_model()

    return _create_conversational_chain(vectorstore, model, prompt_template_dir, num_docs, verbose)


def chat(docs_dirs: tuple[str],
         embeddings_model: str,
         prompt_template_dir: str,
         num_docs: int,
         overwrite_cached_embeddings: bool,
         run_psql_instance: bool,
         verbose: bool):
    conversation_chain = _load_conversational_chain(docs_dirs,
                                                    embeddings_model,
                                                    prompt_template_dir,
                                                    num_docs,
                                                    overwrite_cached_embeddings,
                                                    run_psql_instance,
                                                    verbose)

    prompt = input('> ')
    while prompt != 'exit':
        response = conversation_chain.invoke({'question': prompt})['answer']
        print(response)
        prompt = input('> ')
