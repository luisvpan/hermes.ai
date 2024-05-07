import os

import click

from .model.builder import ModelBuilder

@click.group()
def hermes() -> None:
    pass

@hermes.command('build')
def build_model() -> None:
    model = ModelBuilder().build_model()
    model.save(os.path.join(os.path.dirname(__file__), './model/hermes.keras'))