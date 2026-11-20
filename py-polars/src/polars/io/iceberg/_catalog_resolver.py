from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polars._utils.logging import eprint, verbose
from polars._utils.various import qualified_type_name
from polars.exceptions import ComputeError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    import pyiceberg.catalog

    from polars.io.cloud.credential_provider._providers import (
        CredentialProviderFunction,
    )
    from polars.io.iceberg._dataset import IcebergCatalogConfig

    IcebergCatalogResolver: TypeAlias = Callable[
        [str], "pyiceberg.catalog.Catalog | IcebergResolvedCatalog"
    ]


@dataclass(frozen=True)
class IcebergResolvedCatalog:
    """
    A catalog with the credential provider for its tables' data files.

    Not public API: a resolver may return this instead of a bare catalog. Scans and
    sinks of tables loaded through the catalog's name then read and write data files
    with `credential_provider`, rather than with credentials from the table's FileIO
    properties. Storage options with credentials passed to the scan or sink still
    take precedence.
    """

    catalog: pyiceberg.catalog.Catalog
    credential_provider: CredentialProviderFunction | None = None


# `None` is the default resolver, `pyiceberg.catalog.load_catalog(name)`.
ICEBERG_CATALOG_RESOLVER: IcebergCatalogResolver | None = None


def set_iceberg_catalog_resolver(resolver: IcebergCatalogResolver | None) -> None:
    """
    Set the function that turns an Iceberg catalog name into a catalog.

    Not public API: Polars Cloud installs one on its workers. The resolver returns
    a PyIceberg catalog, or an `IcebergResolvedCatalog` to also give the credential
    provider for the data files of the catalog's tables.

    A catalog name is resolved with the resolver set in the process that loads
    the table, which may be another process than the one that built the plan.
    That process may call it from several threads at once, and more than once
    for the same name; caching catalogs and refreshing their credentials is up to
    the resolver. `None` restores the default resolver.
    """
    global ICEBERG_CATALOG_RESOLVER

    if resolver is not None and not callable(resolver):
        msg = (
            "the Iceberg catalog resolver must be callable or None; found "
            f"{qualified_type_name(resolver)!r}"
        )
        raise TypeError(msg)

    ICEBERG_CATALOG_RESOLVER = resolver


def check_catalog_name(name: str) -> str:
    if not name:
        msg = "Iceberg catalog name must not be empty"
        raise ValueError(msg)
    return name


def resolve_catalog(name: str) -> pyiceberg.catalog.Catalog:
    """Turn a catalog name into a PyIceberg catalog, with the current resolver."""
    return resolve_catalog_with_credential_provider(name).catalog


def resolve_catalog_with_credential_provider(name: str) -> IcebergResolvedCatalog:
    """As `resolve_catalog`, with the resolver's credential provider, if any."""
    import pyiceberg.catalog
    from pyiceberg.catalog.noop import NoopCatalog

    resolver = ICEBERG_CATALOG_RESOLVER
    credential_provider: CredentialProviderFunction | None = None

    if verbose():
        eprint(
            f"resolve_catalog: {name!r} with "
            f"{'the default' if resolver is None else 'an installed'} resolver"
        )

    if resolver is None:
        try:
            catalog = pyiceberg.catalog.load_catalog(name)
        except Exception as error:
            msg = (
                f"failed to load Iceberg catalog {name!r} ({error = }). "
                "Configure it for PyIceberg (a .pyiceberg.yaml file or "
                "PYICEBERG_CATALOG__* environment variables)."
            )
            raise ComputeError(msg) from error
    else:
        # An installed resolver's errors are meant for the user as written; the
        # note says where they come from, as they may surface far from the scan.
        try:
            resolved = resolver(name)
        except Exception as error:
            error.add_note(
                f"raised by the Iceberg catalog resolver for catalog {name!r}"
            )
            raise

        if isinstance(resolved, IcebergResolvedCatalog):
            catalog = resolved.catalog
            credential_provider = resolved.credential_provider
        else:
            catalog = resolved

    if not isinstance(catalog, pyiceberg.catalog.Catalog):
        msg = (
            f"the Iceberg catalog resolver returned {qualified_type_name(catalog)!r} "
            f"for catalog {name!r}; expected a PyIceberg catalog"
        )
        raise TypeError(msg)

    if isinstance(catalog, NoopCatalog):
        msg = (
            f"the Iceberg catalog resolver returned a NoopCatalog for catalog {name!r}"
        )
        raise TypeError(msg)

    return IcebergResolvedCatalog(catalog, credential_provider)


def catalog_from(config: IcebergCatalogConfig | str) -> pyiceberg.catalog.Catalog:
    """A new catalog from its config, or by resolving its name."""
    if isinstance(config, str):
        return resolve_catalog(config)
    return config.class_(config.name, **config.properties)
