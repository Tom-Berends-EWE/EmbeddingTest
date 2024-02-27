__all__ = ['embed_documents', 'discard_embeddings_cache']

import os
import re
from contextlib import nullcontext
from functools import cache
from glob import glob
from hashlib import sha256
from typing import Iterator, Type, Callable, Iterable, Generator

from halo import Halo
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredExcelLoader,
    CSVLoader
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from more_itertools import flatten
from tqdm import tqdm

from aws_embeddings import AWSEmbeddings
from optional_withhold_base_store import OptionalWithholdBaseStore
from util import create_psql_process


def _get_document_loader(file_ext: str) -> Type[BaseLoader]:
    match file_ext:
        case '.pdf':
            return PyMuPDFLoader
        case '.txt':
            return TextLoader
        case '.doc' | '.docx':
            return Docx2txtLoader
        case '.ppt' | '.pptx':
            return UnstructuredPowerPointLoader
        case '.html':
            return UnstructuredHTMLLoader
        case '.xls':
            return UnstructuredExcelLoader
        case '.csv':
            return CSVLoader
        case _:
            raise ValueError(f'Unsupported file extension: "{file_ext}"')


def _load_document(path: str) -> list[Document]:
    file_extension: str
    _, file_extension = os.path.splitext(path)

    loader_class: Type[BaseLoader] = _get_document_loader(file_extension)
    document_loader: BaseLoader = loader_class(path)  # noqa

    return document_loader.load()


def _load_documents(document_paths: Iterator[str] | Iterable[str]) -> Iterator[Document]:
    return flatten(map(_load_document, tqdm(document_paths, desc='Loading documents', leave=False, colour='CYAN')))


def _split_documents(documents: Iterator[Document]) -> list[Document]:
    text_splitter: TextSplitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    return text_splitter.split_documents(documents)


def _postprocess_split_documents(documents: list[Document]) -> None:
    for doc in documents:
        doc.page_content = re.sub(r'[^\x20-\x7E]', '', doc.page_content)


def _create_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    # noinspection SpellCheckingInspection
    model_path = 'intfloat/multilingual-e5-large'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def _create_aws_embeddings() -> AWSEmbeddings:
    return AWSEmbeddings(os.environ['AWS_API_URL'])


_AVAILABLE_EMBEDDINGS_MODELS = {
    'LOCAL': _create_hugging_face_embeddings,
    'HUGGING_FACE': _create_hugging_face_embeddings,
    'AWS': _create_aws_embeddings
}


def _select_embeddings_model(embeddings_model: str) -> Embeddings:
    embeddings_model = embeddings_model.upper().replace('-', '_').replace(' ', '_')

    if embeddings_model not in _AVAILABLE_EMBEDDINGS_MODELS.keys():
        raise ValueError(f'"{embeddings_model}" is not an available embeddings model.')

    embeddings_factory = _AVAILABLE_EMBEDDINGS_MODELS[embeddings_model]
    return embeddings_factory()


@cache
def _get_cache_file_store():
    return LocalFileStore('./cache/embeddings/')


def _delete_from_cache_where(__filter: Callable[[str], bool]):
    store = _get_cache_file_store()
    store.mdelete(filter(__filter, store.yield_keys()))


def _create_embeddings(embeddings_model: str, overwrite_cached_embeddings: bool, namespace: str) -> Embeddings:
    underlying_embeddings = _select_embeddings_model(embeddings_model)

    store = _get_cache_file_store()

    if overwrite_cached_embeddings:
        store = OptionalWithholdBaseStore.always_withhold(store)

    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=namespace
    )


def _compute_vectorstore_pgvector(documents: list[Document], embeddings: Embeddings) -> VectorStore:
    db_connection_string = PGVector.connection_string_from_db_params(
        driver=os.environ['PGVECTOR_DRIVER'],
        host=os.environ['PGVECTOR_HOST'],
        port=int(os.environ['PGVECTOR_PORT']),
        database=os.environ['PGVECTOR_DATABASE'],
        user=os.environ['PGVECTOR_USER'],
        password=os.environ['PGVECTOR_PASSWORD']
    )

    return PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name='compliance_embeddings',
        connection_string=db_connection_string,
        distance_strategy=DistanceStrategy.COSINE
    )


def _compute_vectorstore_faiss(documents: list[Document], embeddings: Embeddings) -> VectorStore:
    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )


def _compute_vectorstore(documents: list[Document], embeddings: Embeddings) -> VectorStore:
    return _compute_vectorstore_faiss(documents, embeddings)


@cache
def _uuid_str(s: str) -> str:
    return str(int(sha256(s.encode('utf-8')).hexdigest(), 16))


def discard_embeddings_cache(discard_all: bool, exclude_selected: bool, embeddings_models: tuple[str]) -> None:
    def __filter(filename: str) -> bool:
        return (discard_all or not (exclude_selected or embeddings_models) or
                any(
                    exclude_selected != filename.startswith(_uuid_str(model))
                    for model in embeddings_models
                ))

    _delete_from_cache_where(__filter)


def _get_source_documents(docs_dirs: Iterator[str] | Iterable[str]) -> Iterable[str]:
    return set(flatten(glob(os.path.abspath(docs_dir) + '/**/*.*', recursive=True) for docs_dir in docs_dirs))


def _generate_embeddings(halo: Halo,
                         documents: list[Document],
                         embeddings_models: tuple[str],
                         overwrite_cached_embeddings: bool) -> Generator[VectorStore, None, None]:
    for embeddings_model in embeddings_models:
        halo.text = f'Generating embeddings using model "{embeddings_model}"'
        try:
            embeddings = _create_embeddings(embeddings_model,
                                            overwrite_cached_embeddings,
                                            _uuid_str(embeddings_model))
        except ValueError as err:
            halo.warn(str(err))
            continue

        yield _compute_vectorstore(documents, embeddings)


def embed_documents(docs_dirs: tuple[str],
                    embeddings_models: tuple[str],
                    overwrite_cached_embeddings: bool,
                    run_psql_instance: bool) -> list[VectorStore]:
    source_documents: Iterable[str] = _get_source_documents(docs_dirs)
    documents: Iterator[Document] = _load_documents(source_documents)
    split_documents: list[Document] = _split_documents(documents)
    _postprocess_split_documents(split_documents)

    with (
        Halo(text='Loading embeddings...', spinner='bouncingBar') as halo,
        create_psql_process() if run_psql_instance else nullcontext(),
    ):
        return [*_generate_embeddings(halo, split_documents, embeddings_models, overwrite_cached_embeddings)]
