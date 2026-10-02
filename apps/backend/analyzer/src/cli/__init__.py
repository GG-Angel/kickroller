import typer

app = typer.Typer()


@app.command("analyze")
def analyze():
    pass


@app.command("train")
def train():
    pass


@app.command("generate")
def generate():
    pass
