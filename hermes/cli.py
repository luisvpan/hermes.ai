import os

import click
import tkinter as tk

from .model.super_hermes_builder import ModelBuilder

@click.group()
def hermes() -> None:
    pass

@hermes.command('build')
def build_model() -> None:
    model = ModelBuilder().build_model()
    model.save(os.path.join(os.path.dirname(__file__), './model/super_hermes.keras'))


@hermes.command('predict')
def predict() -> None:
    root = tk.Tk()
    WhiteboardApp(root)
    root.mainloop()
