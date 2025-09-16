#!/usr/bin/env nix-shell
#! nix-shell -i python3 -p python313Packages.numpy python313Packages.pandas python313Packages.click
from math import sqrt

import click
import numpy as np
import pandas as pd


def _create_contingency_table(cohort):
    return pd.crosstab(
        (cohort["hair_color"].str.lower() == "black").astype(int),
        (cohort["eye_color"].str.lower() == "brown").astype(int),
    )


def _calculate_phi_from_cells(a, b, c, d):
    denominator = sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denominator if denominator != 0 else 0


def generate_cohort_with_phi(n, target_phi, p_black_hair=0.5, p_brown_eyes=0.5):
    n_black_hair = int(n * p_black_hair)
    n_brown_eyes = int(n * p_brown_eyes)

    d_independent = (n_black_hair * n_brown_eyes) / n
    d = max(
        0,
        min(
            int(
                round(
                    d_independent
                    + target_phi
                    * sqrt(
                        n_black_hair
                        * (n - n_black_hair)
                        * n_brown_eyes
                        * (n - n_brown_eyes)
                    )
                    / (2 * sqrt(n))
                )
            ),
            min(n_black_hair, n_brown_eyes),
        ),
    )

    c = n_black_hair - d
    b = n_brown_eyes - d
    a = n - n_black_hair - b

    if min(a, b, c, d) < 0:
        d = int(round(d_independent))
        c = n_black_hair - d
        b = n_brown_eyes - d
        a = n - n_black_hair - b

    cohort_data = []
    for hair_color, eye_color, count in [
        ("not_black", "not_brown", a),
        ("not_black", "brown", b),
        ("black", "not_brown", c),
        ("black", "brown", d),
    ]:
        cohort_data.extend([{"hair_color": hair_color, "eye_color": eye_color}] * count)

    np.random.shuffle(cohort_data)
    return pd.DataFrame(cohort_data)


def compute_phi_coefficient(cohort):
    table = _create_contingency_table(cohort)
    return _calculate_phi_from_cells(
        table.iloc[0, 0], table.iloc[0, 1], table.iloc[1, 0], table.iloc[1, 1]
    )


@click.group()
def cli():
    """Phi coefficient generator and calculator for binary categorical data."""
    pass


@cli.command()
@click.option("--size", "-n", default=1000, type=int, help="Size of cohort to generate")
@click.option(
    "--phi", "-p", default=0.5, type=float, help="Target phi coefficient (-1 to 1)"
)
@click.option(
    "--p-black-hair",
    default=0.5,
    type=float,
    help="Proportion with black hair (0 to 1)",
)
@click.option(
    "--p-brown-eyes",
    default=0.5,
    type=float,
    help="Proportion with brown eyes (0 to 1)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output CSV file path (prints to stdout if not provided)",
)
def generate(size, phi, p_black_hair, p_brown_eyes, output):
    """Generate a cohort with the specified phi coefficient."""
    cohort = generate_cohort_with_phi(size, phi, p_black_hair, p_brown_eyes)
    actual_phi = compute_phi_coefficient(cohort)

    click.echo(f"Generated cohort with {len(cohort)} individuals")
    click.echo(f"Target phi: {phi:.3f}, Actual phi: {actual_phi:.3f}")

    if output:
        cohort.to_csv(output, index=False)
        click.echo(f"Cohort saved to {output}")
    else:
        click.echo("\nCohort data:")
        click.echo(cohort.to_csv(index=False))


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
def compute(csv_path):
    """Compute phi coefficient for black hair and brown eyes from CSV file."""
    try:
        cohort = pd.read_csv(csv_path)

        required_columns = {"hair_color", "eye_color"}
        if not required_columns.issubset(cohort.columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")

        phi = compute_phi_coefficient(cohort)
        table = _create_contingency_table(cohort)
        table.index = ["Not Black Hair", "Black Hair"]
        table.columns = ["Not Brown Eyes", "Brown Eyes"]

        click.echo(f"Cohort size: {len(cohort)}")
        click.echo(f"Phi coefficient: {phi:.3f}")
        click.echo("\nContingency table:")
        click.echo(table.to_string())

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
