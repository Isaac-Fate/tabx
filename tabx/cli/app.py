import typer


APP_NAME = "TabX"

app = typer.Typer(
    help="A tool for extracting tables from PDFs.",
)


@app.command()
def hello():
    print("Welcome to TabX!")
