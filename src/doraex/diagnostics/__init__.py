"""Reusable diagnostic products for Doraex retrieval runs."""

__all__ = [
    "compare_single_line_pressure_response_main",
    "make_mean_subtracted_line_stack_main",
]


def __getattr__(name):
    """Load CLI entry points lazily so submodules remain executable with -m."""

    if name == "compare_single_line_pressure_response_main":
        from doraex.diagnostics.single_line_pressure_response import main

        return main
    if name == "make_mean_subtracted_line_stack_main":
        from doraex.diagnostics.mean_subtracted_line_stack import main

        return main
    raise AttributeError(name)
