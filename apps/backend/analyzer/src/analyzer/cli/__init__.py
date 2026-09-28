"""The command line interface: one module per command."""

import typer

from analyzer.cli import analyze, synth, train

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def cli() -> None:
    """Find the kicks and the beat grid of rawstyle tracks."""


app.command()(analyze.analyze)
app.command()(train.train)
app.command()(synth.synth)


def main() -> None:
    app()
