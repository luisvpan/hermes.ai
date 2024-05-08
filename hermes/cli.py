import os

import click
import tkinter as tk

from .model.builder import ModelBuilder
from .board import WhiteboardApp

@click.group()
def hermes() -> None:
    pass

@hermes.command('build')
def build_model() -> None:
    model = ModelBuilder().build_model()
    model.save(os.path.join(os.path.dirname(__file__), './model/hermes.keras'))


@hermes.command('predict')
def predict() -> None:
    root = tk.Tk()
    WhiteboardApp(root)
    root.mainloop()