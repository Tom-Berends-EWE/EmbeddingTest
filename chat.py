__all__ = ['chat']

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

_HUMAN_MESSAGE_TEMPLATE = '{question}'


def _load_prompt_template(path: str) -> str:
    with open(path, 'r') as f:
        return '\n'.join(f.readlines())


def _create_model() -> BaseChatModel:
    return ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k')


def _create_conversational_chain(vectorstore: VectorStore,
                                 model: BaseChatModel,
                                 system_message_prompt_template: str,
                                 num_docs: int,
                                 verbose: bool) -> BaseConversationalRetrievalChain:
    system_message_prompt = SystemMessagePromptTemplate.from_template(template=system_message_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(template=_HUMAN_MESSAGE_TEMPLATE)
    question_answering_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    condense_question_prompt = PromptTemplate.from_template(
        _load_prompt_template('res/templates/condense-question-prompt-template.txt')
    )

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
                               system_message_prompt_template_path: str,
                               num_docs: int,
                               overwrite_cached_embeddings: bool,
                               run_psql_instance: bool,
                               verbose: bool) -> BaseConversationalRetrievalChain:
    generated_vectorstores: list[VectorStore] = embed_documents(docs_dirs,
                                                                (embeddings_model,),
                                                                overwrite_cached_embeddings,
                                                                run_psql_instance)
    if not generated_vectorstores:
        print(f'Could not generate embeddings with model "{embeddings_model}"!')
        exit(-1)

    vectorstore: VectorStore = generated_vectorstores[0]
    model: BaseChatModel = _create_model()
    system_message_prompt_template = _load_prompt_template(system_message_prompt_template_path)

    return _create_conversational_chain(vectorstore, model, system_message_prompt_template, num_docs, verbose)


def chat(docs_dirs: tuple[str],
         embeddings_model: str,
         system_message_prompt_template_path: str,
         num_docs: int,
         overwrite_cached_embeddings: bool,
         run_psql_instance: bool,
         verbose: bool):
    conversation_chain = _load_conversational_chain(docs_dirs,
                                                    embeddings_model,
                                                    system_message_prompt_template_path,
                                                    num_docs,
                                                    overwrite_cached_embeddings,
                                                    run_psql_instance,
                                                    verbose)

    prompt = input('> ')
    while prompt != 'exit':
        response = conversation_chain.invoke({'question': prompt})['answer']
        print(response)
        prompt = input('> ')
