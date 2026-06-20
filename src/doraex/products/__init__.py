"""Product definitions and post-processing entry points for Doraex runs."""

__all__ = [
    "DEFAULT_PRIMARY_PRODUCT_DEFINITIONS",
    "PrimaryProductConfig",
    "PrimaryProductDefinition",
    "PrimaryProductResult",
    "generate_primary_products",
    "primary_product_definitions",
    "primary_product_paths",
]


def __getattr__(name):
    """Load product helpers lazily to keep imports lightweight."""

    if name in __all__:
        from doraex.products.primary import (
            DEFAULT_PRIMARY_PRODUCT_DEFINITIONS,
            PrimaryProductConfig,
            PrimaryProductDefinition,
            PrimaryProductResult,
            generate_primary_products,
            primary_product_definitions,
            primary_product_paths,
        )

        return {
            "DEFAULT_PRIMARY_PRODUCT_DEFINITIONS": DEFAULT_PRIMARY_PRODUCT_DEFINITIONS,
            "PrimaryProductConfig": PrimaryProductConfig,
            "PrimaryProductDefinition": PrimaryProductDefinition,
            "PrimaryProductResult": PrimaryProductResult,
            "generate_primary_products": generate_primary_products,
            "primary_product_definitions": primary_product_definitions,
            "primary_product_paths": primary_product_paths,
        }[name]
    raise AttributeError(name)
