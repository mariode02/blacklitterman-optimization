import typer
from bl_optimizer.cli.commands import run, validate, backtest

app = typer.Typer(
    name="bl-optimize",
    help="Black-Litterman portfolio optimization with ML-generated views.",
    add_completion=False,
)

app.add_typer(run.app, name="run")
app.command("validate")(validate.validate_command)
app.command("backtest")(backtest.backtest_command)


def main() -> None:
    """Main entry point for the bl-optimize CLI."""
    app()

if __name__ == "__main__":
    main()
