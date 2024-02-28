__all__ = ['retrieve']

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from embedding import embed_documents


def _print_document(doc: Document, verbose: bool) -> None:
    print(f'### DOCUMENT ###\n{doc.page_content}')

    if verbose:
        print(f'### METADATA ###\n{doc.metadata}')

    print('### DOCUMENT END ###\n')


def retrieve(docs_dirs: tuple[str],
             embeddings_model: str,
             num_docs: int,
             overwrite_cached_embeddings: bool,
             run_psql_instance: bool,
             verbose: bool) -> None:
    vectorstore: VectorStore = embed_documents(
        docs_dirs,
        (embeddings_model,),
        overwrite_cached_embeddings,
        run_psql_instance,
        exit_if_unknown_model=True
    )[0]

    prompt = input('> ')
    while prompt != 'exit':
        returned_docs: list[Document] = vectorstore.similarity_search(query=prompt, k=num_docs)
        for doc in returned_docs:
            _print_document(doc, verbose)
        prompt = input('> ')
