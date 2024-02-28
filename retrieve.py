__all__ = ['retrieve']

from langchain_core.vectorstores import VectorStore

from embedding import embed_documents


def retrieve(docs_dirs: tuple[str],
             embeddings_model: str,
             num_docs: int,
             overwrite_cached_embeddings: bool,
             run_psql_instance: bool) -> None:
    vectorstore: VectorStore = embed_documents(
        docs_dirs,
        (embeddings_model,),
        overwrite_cached_embeddings,
        run_psql_instance,
        exit_if_unknown_model=True
    )[0]

    prompt = input('> ')
    while prompt != 'exit':
        returned_docs = vectorstore.similarity_search(query=prompt, k=num_docs)
        for doc in returned_docs:
            print(f'\n###\n{doc.page_content}\n###\n')
        prompt = input('> ')
