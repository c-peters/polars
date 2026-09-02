use std::sync::Arc;

use polars::prelude::*;
use polars_compute::bloom_filter::{insert, num_bytes_for};
use polars_plan::dsl::{BooleanFunction, FunctionExpr};

/// `IsInBloomFilter` has no public DSL constructor -- it is inserted into the
/// IR by the distributed engine. Build it by hand so the whole pipeline
/// (dsl -> ir -> physical expr -> execution) is exercised.
fn is_in_bloom_filter(input: Expr, bitset: Vec<u8>) -> Expr {
    Expr::Function {
        input: vec![input],
        function: FunctionExpr::Boolean(BooleanFunction::IsInBloomFilter {
            bitset: Arc::from(bitset.into_boxed_slice()),
        }),
    }
}

fn filter_over(values: &[u64]) -> Vec<u8> {
    let mut bitset = vec![0u8; num_bytes_for(values.len().max(1), 0.01, 1 << 16)];
    for v in values {
        insert(&mut bitset, *v);
    }
    bitset
}

/// The keys the build side of a join would have produced.
const BUILD_KEYS: [u64; 4] = [10, 20, 30, 40];

fn probe_frame() -> LazyFrame {
    df!["k" => [10u64, 11, 20, 21, 30, 31, 40, 41]]
        .unwrap()
        .lazy()
}

#[test]
fn filters_probe_side_without_dropping_matches() {
    let bitset = filter_over(&BUILD_KEYS);

    let out = probe_frame()
        .filter(is_in_bloom_filter(col("k"), bitset))
        .collect()
        .unwrap();

    let kept: Vec<u64> = out
        .column("k")
        .unwrap()
        .u64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    // Every real match must survive. The filter is allowed to let extra rows
    // through (false positives), but never to drop a true match.
    for key in BUILD_KEYS {
        assert!(kept.contains(&key), "dropped a true match: {key}");
    }
    // With a filter this small over 4 values, nothing else should slip through.
    assert_eq!(kept, BUILD_KEYS.to_vec());
}

/// The point of the optimization: pre-filtering the probe side must not change
/// the result of the join it feeds.
#[test]
fn prefiltering_does_not_change_join_result() {
    let build = df!["k" => BUILD_KEYS.to_vec(), "v" => ["a", "b", "c", "d"]]
        .unwrap()
        .lazy();
    let bitset = filter_over(&BUILD_KEYS);

    let unfiltered = probe_frame()
        .join(
            build.clone(),
            [col("k")],
            [col("k")],
            JoinArgs::new(JoinType::Inner),
        )
        .unwrap()
        .sort(["k"], Default::default())
        .collect()
        .unwrap();

    let prefiltered = probe_frame()
        .filter(is_in_bloom_filter(col("k"), bitset))
        .join(
            build,
            [col("k")],
            [col("k")],
            JoinArgs::new(JoinType::Inner),
        )
        .unwrap()
        .sort(["k"], Default::default())
        .collect()
        .unwrap();

    assert_eq!(unfiltered, prefiltered);
}

#[test]
fn nulls_do_not_survive_the_filter() {
    let bitset = filter_over(&BUILD_KEYS);

    let out = df!["k" => [Some(10u64), None, Some(11)]]
        .unwrap()
        .lazy()
        .filter(is_in_bloom_filter(col("k"), bitset))
        .collect()
        .unwrap();

    assert_eq!(out.height(), 1);
    assert_eq!(out.column("k").unwrap().u64().unwrap().get(0), Some(10));
}

/// The distributed engine puts this expression on a scan's predicate, so
/// pushdown has to accept it and the scan has to evaluate it. The streaming
/// parquet reader then decodes only the key column before filtering.
#[test]
fn pushes_down_into_a_parquet_scan() {
    let dir = std::env::temp_dir().join(format!(
        "polars-bloom-filter-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("keys.parquet");

    let mut df = df![
        "k" => [10u64, 11, 20, 21, 30, 31, 40, 41],
        "v" => [0i64, 1, 2, 3, 4, 5, 6, 7],
    ]
    .unwrap();
    let file = std::fs::File::create(&path).unwrap();
    ParquetWriter::new(file).finish(&mut df).unwrap();

    let bitset = filter_over(&BUILD_KEYS);
    let lf = LazyFrame::scan_parquet(PlRefPath::new(path.to_str().unwrap()), Default::default())
        .unwrap()
        .filter(is_in_bloom_filter(col("k"), bitset));

    // The predicate has to reach the scan; a `FILTER` above it would mean
    // pushdown rejected the expression and the reader decodes every column.
    let plan = lf.clone().explain(true).unwrap();
    assert!(
        plan.contains("SELECTION") && plan.contains("is_in_bloom_filter"),
        "predicate did not reach the scan:\n{plan}"
    );

    let out = lf.collect().unwrap();
    let kept: Vec<u64> = out
        .column("k")
        .unwrap()
        .u64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(kept, BUILD_KEYS.to_vec());

    std::fs::remove_dir_all(&dir).ok();
}
