__all__ = ['cli']

import click
from click import option

from chat import chat
from embedding import embed_documents, discard_embeddings_cache
from mutex_option import Mutex

loading_directory_option = option(
    '--dir',
    '-d',
    'docs_dirs',
    multiple=True,
    default=('res/docs/', )
)
overwrite_cache_option = option(
    '--overwrite-cached-embeddings/--use-cached-embeddings',
    '-o',
    default=False
)
run_psql_instance_option = option(
    '--run-psql-instance',
    '-r',
    is_flag=True
)


@click.group()
def cli():
    pass


@cli.command(name='discard_cache')
@option(
    '--discard-all',
    '-a',
    is_flag=True,
)
@option(
    '--exclude/--include',
    '-e/-i',
    'exclude_selected',
    cls=Mutex,
    default=False,
    not_required_if=['discard_all']
)
@option(
    '--embeddings-model',
    '-m',
    'embeddings_models',
    cls=Mutex,
    multiple=True,
    not_required_if=['discard_all']
)
def _discard_cache_command(discard_all: bool, exclude_selected: bool, embeddings_models: tuple[str]) -> None:
    discard_embeddings_cache(discard_all, exclude_selected, embeddings_models)


@cli.command(name='embed')
@loading_directory_option
@option(
    '--embeddings-model',
    '-m',
    'embeddings_models',
    multiple=True,
    default=('AWS', )
)
@overwrite_cache_option
@run_psql_instance_option
def _embed_documents_command(docs_dirs: tuple[str],
                             embeddings_models: tuple[str],
                             overwrite_cached_embeddings: bool,
                             run_psql_instance: bool) -> None:
    embed_documents(docs_dirs, embeddings_models, overwrite_cached_embeddings, run_psql_instance)


@cli.command(name='chat')
@loading_directory_option
@option(
    '--embeddings-model',
    '-m',
    default='AWS'
)
@option(
    '--prompt-template-dir',
    '-t'
)
@option(
    '--returned-docs',
    '-k',
    'num_docs',
    default=5,
    type=int
)
@overwrite_cache_option
@run_psql_instance_option
@option(
    '--verbose/--quiet',
    '-v',
    default=False
)
def _chat_command(docs_dirs: tuple[str],
                  embeddings_model: str,
                  prompt_template_dir: str,
                  num_docs: int,
                  overwrite_cached_embeddings: bool,
                  run_psql_instance: bool,
                  verbose: bool) -> None:
    chat(docs_dirs, embeddings_model, prompt_template_dir, num_docs, overwrite_cached_embeddings, run_psql_instance, verbose)
