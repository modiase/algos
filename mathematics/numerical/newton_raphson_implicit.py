from collections.abc import Callable, Mapping, Sequence
from functools import reduce

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def f(x: float) -> float:
    """
    f(x) = x^3 - 2x - 5
    """
    return x**3 - 2 * x - 5


def df_dx(x: float) -> float:
    """
    f'(x) = 3x^2 - 2
    """
    return 3 * x**2 - 2


def problematic_f(x: float) -> float:
    """
    f(x) = x^4 - 10x^2 + 9
    """
    return x**4 - 10 * x**2 + 9


def problematic_df_dx(x: float) -> float:
    """
    f'(x) = 4x^3 - 20x
    """
    return 4 * x**3 - 20 * x


def newton_explicit(
    f: Callable[[float], float],
    df_dx: Callable[[float], float],
    x0: float,
    iterations: int = 100,
) -> Sequence[float]:
    """
    Explicit Newton-Raphson method for finding roots
    """
    return reduce(
        lambda acc, _: acc + [(x := acc[-1]) - f(x) / df_dx(x)],
        range(iterations),
        [x0],
    )


def newton_implicit(
    f: Callable[[float], float],
    df_dx: Callable[[float], float],
    x0: float,
    eta: float = 0.5,
    iterations: int = 100,
    inner_iterations: int = 10,
) -> Sequence[float]:
    """
    Implicit method for finding roots using nested reduce operations

    Instead of $x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}$,

    we find $x_{n+1}$ that satisfies:

    $x_{n+1} = x_n - \\eta * \\frac{f(x_{n+1})}{f'(x_{n+1})}$

    by iterating the following:

    $g(x_{n+1}) = x_{n+1} - x_n + \\eta * \\frac{f(x_{n+1})}{f'(x_{n+1})}$

    $g'(x_{n+1}) = 1 + \\eta * \\frac{(f'(x_{n+1}))^2 - f(x_{n+1}) * f''(x_{n+1})}{(f'(x_{n+1}))^2}$

    $x_{n+1} = x_n - \\frac{g(x_{n+1})}{g'(x_{n+1})}$

    until convergence.
    """
    return reduce(
        lambda acc, _: acc
        + [
            reduce(
                lambda x_inner, _: (
                    fx := f(x_inner),
                    dfx := df_dx(x_inner),
                    g_x := x_inner - acc[-1] + eta * fx / dfx,
                    dg_dx := 1 + eta * (dfx**2 - fx * (6 * x_inner)) / (dfx**2),
                    x_inner - g_x / dg_dx,
                )[-1],
                range(inner_iterations),
                acc[-1],
            )
        ],
        range(iterations),
        [x0],
    )


def plot_convergence(
    iterations: Mapping[str, Sequence[float]],
    root: float,
):
    N = len(iterations)

    fig = make_subplots(
        rows=N + 1,
        cols=1,
        subplot_titles=list(iterations.keys()) + ["Convergence Error Comparison"],
        vertical_spacing=0.05,
        specs=[[{"type": "xy"}] for _ in range(N + 1)],
    )

    colors = (
        plotly.colors.qualitative.Plotly[:N]
        if N <= 10
        else plotly.colors.sequential.Viridis[:N]
    )

    for idx, (label, i) in enumerate(iterations.items()):
        x_vals = list(range(len(i)))

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=i,
                mode="lines+markers",
                name=label,
                line=dict(color=colors[idx], width=3),
                marker=dict(size=10),
                legendgroup=label,
                showlegend=False,
            ),
            row=idx + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[min(x_vals), max(x_vals)],
                y=[root, root],
                mode="lines",
                name=f"True root: {root:.5f}",
                line=dict(color="red", dash="dash", width=3),
                legendgroup=f"root_{idx}",
                showlegend=(idx == 0),
            ),
            row=idx + 1,
            col=1,
        )

        fig.update_xaxes(
            title_text="Iterations",
            row=idx + 1,
            col=1,
            title_font=dict(size=16),
            tickfont=dict(size=14),
        )
        fig.update_yaxes(
            title_text="x value",
            row=idx + 1,
            col=1,
            title_font=dict(size=16),
            tickfont=dict(size=14),
        )

    errors = {label: [abs(x - root) for x in i] for label, i in iterations.items()}

    for idx, (label, e) in enumerate(errors.items()):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(e))),
                y=e,
                mode="lines+markers",
                name=label,
                line=dict(color=colors[idx], width=3),
                marker=dict(size=10),
            ),
            row=N + 1,
            col=1,
        )

    fig.update_yaxes(
        type="log",
        row=N + 1,
        col=1,
        title_text="Absolute Error (log scale)",
        title_font=dict(size=16),
        tickfont=dict(size=14),
    )
    fig.update_xaxes(
        title_text="Iterations",
        row=N + 1,
        col=1,
        title_font=dict(size=16),
        tickfont=dict(size=14),
    )

    subplot_height = 400
    total_height = subplot_height * (N + 1)

    fig.update_layout(
        height=total_height,
        width=1200,
        title_font=dict(size=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=16),
        ),
        margin=dict(l=60, r=60, t=120, b=60),
    )

    for i in range(1, N + 2):
        fig.update_layout(
            {
                f"yaxis{i if i > 1 else ''}": dict(
                    domain=[1 - i / (N + 1), 1 - (i - 1) / (N + 1) - 0.02]
                )
            }
        )

    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=18)

    fig.show(scrollZoom=True)


if __name__ == "__main__":
    x0 = 5
    exact_root = 1.0
    plot_convergence(
        {
            "explicit": newton_explicit(problematic_f, problematic_df_dx, x0),
            "implicit eta 0.1": newton_implicit(
                problematic_f, problematic_df_dx, x0, eta=0.1
            ),
            "implicit eta 1000.0": newton_implicit(
                problematic_f, problematic_df_dx, x0, eta=1000.0
            ),
        },
        exact_root,
    )

    x0 = 1.0
    exact_root = 2.09455148154233
    plot_convergence(
        {
            "explicit": newton_explicit(f, df_dx, x0),
            "implicit eta 0.5": newton_implicit(f, df_dx, x0, eta=0.5),
            "implicit eta 1.0": newton_implicit(f, df_dx, x0, eta=1.0),
            "implicit eta 10.0": newton_implicit(f, df_dx, x0, eta=10.0),
            "implicit eta 100.0": newton_implicit(f, df_dx, x0, eta=100.0),
        },
        exact_root,
    )
