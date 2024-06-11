import os

import click
import tkinter as tk

from .hermes_board import WhiteboardApp
from .model.hermes_builder import ModelBuilder
from .model.super_hermes_builder import ModelBuilder as SuperModelBuilder

@click.group()
def hermes() -> None:
    pass

@hermes.command('build')
def build_model() -> None:
    model = SuperModelBuilder().build_model()
    model.save(os.path.join(os.path.dirname(__file__), './model/super_hermes.keras'))


@hermes.command('predict')
def predict() -> None:
    root = tk.Tk()
    WhiteboardApp(root)
    root.mainloop()