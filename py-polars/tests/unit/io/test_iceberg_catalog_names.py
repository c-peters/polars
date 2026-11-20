from __future__ import annotations

import io
import os
import pickle
import subprocess
import sys
import warnings
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal

pytest.importorskip("pyiceberg.catalog.sql")

import pyiceberg.catalog
from pyiceberg.catalog.noop import NoopCatalog
from pyiceberg.catalog.sql import SqlCatalog

from polars.io.iceberg import _catalog_resolver
from polars.io.iceberg._catalog_resolver import (
    IcebergResolvedCatalog,
    check_catalog_name,
    resolve_catalog,
    resolve_catalog_with_credential_provider,
    set_iceberg_catalog_resolver,
)
from polars.io.iceberg._sink import IcebergSinkState
from tests.unit.io.conftest import (
    failing_catalog_resolver,
    recording_catalog_resolver,
)

pytestmark = [
    pytest.mark.write_disk,
    pytest.mark.usefixtures("reset_iceberg_catalog_resolver"),
]

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pyiceberg.catalog import Catalog


@pytest.fixture(autouse=True)
def sqlcatalog_uses_null_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    # the default pool keeps sqlite connections open past the test (ResourceWarning)
    import pyiceberg.catalog.sql as pyiceberg_sql
    from sqlalchemy.pool import NullPool

    create_engine = pyiceberg_sql.create_engine  # type: ignore[attr-defined]
    monkeypatch.setattr(
        pyiceberg_sql,
        "create_engine",
        lambda url, **kwargs: create_engine(url, **{**kwargs, "poolclass": NullPool}),
    )


def new_catalog(tmp_path: Path, name: str, df: pl.DataFrame) -> SqlCatalog:
    path = tmp_path / name
    path.mkdir()
    catalog = SqlCatalog(
        name, uri=f"sqlite:///{path / 'catalog.db'}", warehouse=f"file://{path}"
    )
    catalog.create_namespace("ns")
    table = catalog.create_table(("ns", "t"), schema=df.to_arrow().schema)
    df.write_iceberg(table, mode="append")
    return catalog


@pytest.fixture
def sales(tmp_path: Path) -> SqlCatalog:
    return new_catalog(tmp_path, "sales", pl.DataFrame({"a": [1, 2]}))


def test_set_resolver_rejects_non_callable() -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)

    with pytest.raises(TypeError, match="must be callable or None; found 'dict'"):
        set_iceberg_catalog_resolver({"sales": None})  # type: ignore[arg-type]
    assert _catalog_resolver.ICEBERG_CATALOG_RESOLVER is failing_catalog_resolver


def test_resolve_catalog_uses_installed_resolver(sales: SqlCatalog) -> None:
    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)

    assert resolve_catalog("sales") is sales
    assert names == ["sales"]


def test_resolver_none_restores_default(
    sales: SqlCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    def load_catalog(name: str, **_: Any) -> Catalog:
        loaded.append(name)
        return sales

    monkeypatch.setattr(pyiceberg.catalog, "load_catalog", load_catalog)
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    set_iceberg_catalog_resolver(None)

    assert resolve_catalog("sales") is sales
    assert loaded == ["sales"]


@pytest.mark.slow
def test_default_resolver_reads_pyiceberg_env(sales: SqlCatalog) -> None:
    # PyIceberg reads its environment when it is imported, so in a new process.
    env = {
        **os.environ,
        "PYICEBERG_CATALOG__SALES__TYPE": "sql",
        "PYICEBERG_CATALOG__SALES__URI": sales.properties["uri"],
        "PYICEBERG_CATALOG__SALES__WAREHOUSE": sales.properties["warehouse"],
    }
    script = (
        "from polars.io.iceberg._catalog_resolver import resolve_catalog\n"
        "table = resolve_catalog('sales').load_table('ns.t')\n"
        "print(table.scan().to_arrow().num_rows)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2"


def test_default_resolver_failure_is_compute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_catalog(name: str, **_: Any) -> Catalog:
        msg = "boom"
        raise ValueError(msg)

    monkeypatch.setattr(pyiceberg.catalog, "load_catalog", load_catalog)

    with pytest.raises(
        ComputeError, match="failed to load Iceberg catalog 'sales'"
    ) as excinfo:
        resolve_catalog("sales")
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_resolver_error_passes_through() -> None:
    def resolver(name: str) -> Catalog:
        msg = "not connected"
        raise ComputeError(msg)

    set_iceberg_catalog_resolver(resolver)

    # pytest matches against the message and its notes, so check each apart.
    with pytest.raises(ComputeError, match="not connected") as excinfo:
        resolve_catalog("sales")
    assert str(excinfo.value) == "not connected"
    assert excinfo.value.__notes__ == [
        "raised by the Iceberg catalog resolver for catalog 'sales'"
    ]


def test_resolver_error_names_the_catalog() -> None:
    # A mapping's `__getitem__` raises a KeyError naming only the key.
    catalogs: dict[str, Catalog] = {}
    set_iceberg_catalog_resolver(catalogs.__getitem__)

    with pytest.raises(KeyError) as excinfo:
        resolve_catalog("sale")
    assert excinfo.value.__notes__ == [
        "raised by the Iceberg catalog resolver for catalog 'sale'"
    ]


def test_resolver_returning_non_catalog_is_type_error() -> None:
    set_iceberg_catalog_resolver(lambda _: object())  # type: ignore[arg-type, return-value]

    with pytest.raises(TypeError, match="returned 'object' for catalog 'sales'"):
        resolve_catalog("sales")


def test_resolver_returning_noop_catalog_is_type_error() -> None:
    set_iceberg_catalog_resolver(lambda _: NoopCatalog("x"))

    with pytest.raises(TypeError, match="NoopCatalog for catalog 'sales'"):
        resolve_catalog("sales")


def test_check_catalog_name_rejects_empty() -> None:
    assert check_catalog_name("sales") == "sales"
    with pytest.raises(ValueError, match="must not be empty"):
        check_catalog_name("")


@pytest.fixture
def _pyiceberg_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    # Results must not depend on whether `polars_iceberg` is installed.
    monkeypatch.setenv("POLARS_ICEBERG_PLANNER", "pyiceberg")


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_does_not_resolve_at_build() -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)

    assert isinstance(pl.scan_iceberg("ns.t", catalog="sales"), pl.LazyFrame)


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_resolves_at_collect(sales: SqlCatalog) -> None:
    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)

    df = pl.scan_iceberg("ns.t", catalog="sales").collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_serializes_only_the_name(sales: SqlCatalog) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    data = pl.scan_iceberg("ns.t", catalog="sales").serialize()

    assert b"sales" in data
    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    df = pl.LazyFrame.deserialize(io.BytesIO(data)).collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_uses_resolver_active_at_collect(sales: SqlCatalog) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    lf = pl.scan_iceberg("ns.t", catalog="sales")

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)

    assert_frame_equal(lf.collect(), pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_two_named_catalogs_in_one_query(
    sales: SqlCatalog, tmp_path: Path
) -> None:
    east = new_catalog(tmp_path, "east", pl.DataFrame({"a": [3]}))
    catalogs = {"sales": sales, "east": east}
    names: list[str] = []

    def resolver(name: str) -> Catalog:
        names.append(name)
        return catalogs[name]

    set_iceberg_catalog_resolver(resolver)
    df = pl.concat(
        [
            pl.scan_iceberg("ns.t", catalog="sales"),
            pl.scan_iceberg("ns.t", catalog="east"),
        ]
    ).collect()

    assert_frame_equal(df.sort("a"), pl.DataFrame({"a": [1, 2, 3]}))
    assert set(names) == {"sales", "east"}


@pytest.mark.usefixtures("_pyiceberg_planner")
@pytest.mark.parametrize(
    "load",
    [pl.LazyFrame.collect_schema, pl.LazyFrame.explain, pl.LazyFrame.collect],
    ids=["collect_schema", "explain", "collect"],
)
def test_scan_serializes_only_the_name_after_loading_the_table(
    sales: SqlCatalog, load: Callable[[pl.LazyFrame], Any]
) -> None:
    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    lf = pl.scan_iceberg("ns.t", catalog="sales")
    load(lf)
    assert names == ["sales"]

    # The table loaded here is not pinned: no metadata path replaces the name.
    data = lf.serialize()
    assert b"metadata.json" not in data
    assert b"sqlite" not in data

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    df = pl.LazyFrame.deserialize(io.BytesIO(data)).collect()

    assert_frame_equal(df, pl.DataFrame({"a": [1, 2]}))
    assert names == ["sales"]


def test_scan_empty_catalog_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        pl.scan_iceberg("ns.t", catalog="")


def test_scan_catalog_name_with_static_path(sales: SqlCatalog) -> None:
    metadata_location = sales.load_table("ns.t").metadata_location

    with pytest.raises(TypeError, match="requires a table identifier"):
        pl.scan_iceberg(metadata_location, catalog="sales")


def test_scan_catalog_name_with_table_source(sales: SqlCatalog) -> None:
    with pytest.raises(TypeError, match="requires a table identifier"):
        pl.scan_iceberg(sales.load_table("ns.t"), catalog="sales")


def table_rows(catalog: SqlCatalog) -> pl.DataFrame:
    return pl.scan_iceberg(catalog.load_table("ns.t")).collect().sort("a")


def test_sink_state_does_not_resolve_at_build() -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)

    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")

    assert isinstance(state.attach_sink(pl.LazyFrame({"a": [3]})), pl.LazyFrame)


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_iceberg_appends_through_resolver(sales: SqlCatalog) -> None:
    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)

    pl.LazyFrame({"a": [3]}).sink_iceberg("ns.t", mode="append", catalog="sales")

    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2, 3]}))
    assert "sales" in names


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_plan_serializes_only_the_name(sales: SqlCatalog) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    data = state.attach_sink(pl.LazyFrame({"a": [3]})).serialize()

    assert b"sales" in data
    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    pl.LazyFrame.deserialize(io.BytesIO(data)).collect()

    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2, 3]}))
    assert names


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_surfaces_resolver_error_after_deserialize(sales: SqlCatalog) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    data = state.attach_sink(pl.LazyFrame({"a": [3]})).serialize()

    def refusing(name: str) -> Catalog:
        msg = "can only be resolved while planning"
        raise ComputeError(msg)

    set_iceberg_catalog_resolver(refusing)

    with pytest.raises(
        ComputeError, match="can only be resolved while planning"
    ) as excinfo:
        pl.LazyFrame.deserialize(io.BytesIO(data)).collect()
    assert str(excinfo.value) == "can only be resolved while planning"
    assert excinfo.value.__notes__ == [
        "raised by the Iceberg catalog resolver for catalog 'sales'"
    ]
    assert_frame_equal(table_rows(sales), pl.DataFrame({"a": [1, 2]}))


def test_sink_state_for_catalog_name_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `sink_iceberg` is unstable as a whole and warns so itself.
    monkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IcebergSinkState.new("ns.t", mode="append", catalog="sales")


@pytest.mark.parametrize(
    "field", ["py_catalog_class_module", "py_catalog_class_qualname"]
)
def test_sink_state_rejects_half_set_catalog_class(field: str) -> None:
    set_iceberg_catalog_resolver(failing_catalog_resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    setattr(state, field, "SqlCatalog")

    with pytest.raises(ValueError, match="must both be set, or both be None"):
        state.table()


def test_sink_catalog_name_with_table_target(sales: SqlCatalog) -> None:
    with pytest.raises(TypeError, match="requires a table identifier"):
        IcebergSinkState.new(sales.load_table("ns.t"), catalog="sales")


def test_sink_empty_catalog_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        IcebergSinkState.new("ns.t", catalog="")


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_state_resolves_again_after_pickling(sales: SqlCatalog) -> None:
    # What a commit task in another process receives: the state after planning.
    resolver, _ = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    state._attach_resolved_sink(pl.LazyFrame({"a": [3]})._ldf)
    data = pickle.dumps(state)

    assert b"sqlite" not in data
    assert b"SqlCatalog" not in data

    resolver, names = recording_catalog_resolver(sales)
    set_iceberg_catalog_resolver(resolver)
    pickle.loads(data).table()
    assert names == ["sales"]

    def refusing(name: str) -> Catalog:
        msg = "can only be resolved while planning"
        raise ComputeError(msg)

    set_iceberg_catalog_resolver(refusing)
    with pytest.raises(
        ComputeError, match="can only be resolved while planning"
    ) as excinfo:
        pickle.loads(data).commit([])
    assert str(excinfo.value) == "can only be resolved while planning"
    assert excinfo.value.__notes__ == [
        "raised by the Iceberg catalog resolver for catalog 'sales'"
    ]


def test_resolver_may_return_a_credential_provider(sales: SqlCatalog) -> None:
    def provider() -> tuple[dict[str, str], int | None]:
        return {}, None

    set_iceberg_catalog_resolver(
        lambda _: IcebergResolvedCatalog(sales, credential_provider=provider)
    )

    assert resolve_catalog("sales") is sales
    assert resolve_catalog_with_credential_provider("sales") == (
        IcebergResolvedCatalog(sales, provider)
    )

    set_iceberg_catalog_resolver(lambda _: sales)
    assert resolve_catalog_with_credential_provider("sales") == (
        IcebergResolvedCatalog(sales, None)
    )


@pytest.fixture
def vended_sales(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SqlCatalog:
    """A catalog whose empty table `ns.t` is on S3, with vended credentials."""
    path = tmp_path / "vended"
    path.mkdir()
    catalog = SqlCatalog(
        "sales",
        uri=f"sqlite:///{path / 'catalog.db'}",
        warehouse=f"file://{path}",
        **{
            "s3.region": "eu-west-1",
            "s3.access-key-id": "vended-key",
            "s3.secret-access-key": "vended-secret",
            "s3.session-token": "vended-token",
        },
    )
    catalog.create_namespace("ns")
    table = catalog.create_table(
        ("ns", "t"), schema=pl.Schema({"a": pl.Int64}).to_arrow()
    )
    table.metadata_location = "s3://bucket/t/metadata/v1.metadata.json"
    table.metadata = table.metadata.model_copy(update={"location": "s3://bucket/t"})
    monkeypatch.setattr(catalog, "load_table", lambda identifier: table)
    return catalog


ROLE_CREDENTIALS = {
    "aws_access_key_id": "role-key",
    "aws_secret_access_key": "role-secret",
    "aws_session_token": "role-token",
}


class RoleCredentialProvider:
    """Stands in for e.g. `pl.CredentialProviderAWS(assume_role=...)`."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> tuple[dict[str, str], int | None]:
        self.calls += 1
        return ROLE_CREDENTIALS, None


def native_scan_data(storage_options: dict[str, str] | None = None) -> Any:
    from polars.io.cloud._utils import NoPickleOption
    from polars.io.iceberg._dataset import (
        IcebergCatalogTableDescriptor,
        IcebergScanResolver,
        IcebergScanTableSerializer,
        IcebergTableWrap,
        _NativeIcebergScanData,
    )

    resolver = IcebergScanResolver(
        table=IcebergTableWrap(
            table_=NoPickleOption(None),
            table_descriptor_=IcebergCatalogTableDescriptor(
                table_identifier="ns.t", catalog_config="sales"
            ),
            serializer=IcebergScanTableSerializer(),
            iceberg_storage_properties=storage_options,
        ),
        snapshot_id=None,
        from_snapshot_id_exclusive=None,
        to_snapshot_id_inclusive=None,
        reader_override=None,
        use_metadata_statistics=True,
        fast_deletion_count=False,
        use_pyiceberg_filter=True,
    )
    scan_data = resolver._to_dataset_scan_impl()
    assert isinstance(scan_data, _NativeIcebergScanData)
    return scan_data


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_without_resolver_credential_provider_uses_vended_credentials(
    vended_sales: SqlCatalog,
) -> None:
    set_iceberg_catalog_resolver(lambda _: vended_sales)

    scan_data = native_scan_data()

    assert scan_data.storage_options == {
        "aws_region": "eu-west-1",
        "aws_access_key_id": "vended-key",
        "aws_secret_access_key": "vended-secret",
        "aws_session_token": "vended-token",
    }
    assert scan_data.credential_provider == "auto"


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_uses_resolver_credential_provider(vended_sales: SqlCatalog) -> None:
    from polars.io.cloud.credential_provider._builder import (
        CredentialProviderBuilder,
    )

    provider = RoleCredentialProvider()
    set_iceberg_catalog_resolver(
        lambda _: IcebergResolvedCatalog(vended_sales, credential_provider=provider)
    )

    scan_data = native_scan_data()

    # The catalog's credentials are dropped, its other options kept.
    assert scan_data.storage_options == {"aws_region": "eu-west-1"}
    builder = scan_data.credential_provider
    assert isinstance(builder, CredentialProviderBuilder)
    assert builder.build_credential_provider()() == (ROLE_CREDENTIALS, None)  # type: ignore[misc]
    assert provider.calls == 1

    # Carried as a provider, never as fixed credentials.
    pickled = pickle.dumps(builder)
    assert b"vended-key" not in pickled
    assert b"role-key" not in pickled


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_binds_aws_options_to_resolver_credential_provider(
    vended_sales: SqlCatalog,
) -> None:
    assume_role: Any = {"RoleArn": "arn:aws:iam::123456789012:role/sales"}
    provider = pl.CredentialProviderAWS(assume_role=assume_role)
    set_iceberg_catalog_resolver(
        lambda _: IcebergResolvedCatalog(vended_sales, credential_provider=provider)
    )

    # Options other than credentials still apply.
    scan_data = native_scan_data({"s3.region": "us-east-1"})

    assert scan_data.storage_options == {"aws_region": "us-east-1"}
    bound = scan_data.credential_provider.credential_provider_init.credential_provider
    assert isinstance(bound, pl.CredentialProviderAWS)
    assert bound._storage_options == {"region": "us-east-1"}
    assert b"arn:aws:iam::123456789012:role/sales" in pickle.dumps(
        scan_data.credential_provider
    )


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_scan_user_credentials_win_over_resolver_credential_provider(
    vended_sales: SqlCatalog,
) -> None:
    provider = RoleCredentialProvider()
    set_iceberg_catalog_resolver(
        lambda _: IcebergResolvedCatalog(vended_sales, credential_provider=provider)
    )

    scan_data = native_scan_data(
        {"s3.access-key-id": "user-key", "s3.secret-access-key": "user-secret"}
    )

    assert scan_data.storage_options == {
        "aws_region": "eu-west-1",
        "aws_access_key_id": "user-key",
        "aws_secret_access_key": "user-secret",
    }
    assert scan_data.credential_provider == "auto"
    assert provider.calls == 0


class _SinkArgs(Exception):
    pass


def sink_parquet_kwargs(
    monkeypatch: pytest.MonkeyPatch, storage_options: dict[str, str] | None = None
) -> dict[str, Any]:
    """Keyword arguments a sink to `sales` passes to `sink_parquet`."""

    def capture(self: pl.LazyFrame, *args: Any, **kwargs: Any) -> Any:
        raise _SinkArgs(kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", capture)
    with pytest.raises(_SinkArgs) as e:
        pl.LazyFrame({"a": [1]}).sink_iceberg(
            "ns.t", mode="append", catalog="sales", storage_options=storage_options
        )
    kwargs: dict[str, Any] = e.value.args[0]
    return kwargs


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_uses_resolver_credential_provider(
    vended_sales: SqlCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = RoleCredentialProvider()
    set_iceberg_catalog_resolver(
        lambda _: IcebergResolvedCatalog(vended_sales, credential_provider=provider)
    )

    kwargs = sink_parquet_kwargs(monkeypatch)

    assert kwargs["storage_options"] == {"aws_region": "eu-west-1"}
    built = kwargs["credential_provider"].build_credential_provider()
    assert built() == (ROLE_CREDENTIALS, None)
    assert provider.calls == 1

    # The user's credentials win.
    kwargs = sink_parquet_kwargs(
        monkeypatch,
        {"s3.access-key-id": "user-key", "s3.secret-access-key": "user-secret"},
    )
    assert kwargs["storage_options"] == {
        "aws_region": "eu-west-1",
        "aws_access_key_id": "user-key",
        "aws_secret_access_key": "user-secret",
    }
    assert kwargs["credential_provider"] == "auto"


@pytest.mark.usefixtures("_pyiceberg_planner")
def test_sink_state_does_not_pickle_resolver_credential_provider(
    vended_sales: SqlCatalog,
) -> None:
    provider = RoleCredentialProvider()
    set_iceberg_catalog_resolver(
        lambda _: IcebergResolvedCatalog(vended_sales, credential_provider=provider)
    )
    state = IcebergSinkState.new("ns.t", mode="append", catalog="sales")
    assert state._catalog_credential_provider() is provider

    # Resolved again where the state is unpickled, e.g. for the commit.
    other = RoleCredentialProvider()
    set_iceberg_catalog_resolver(
        lambda _: IcebergResolvedCatalog(vended_sales, credential_provider=other)
    )
    assert pickle.loads(pickle.dumps(state))._catalog_credential_provider() is other
