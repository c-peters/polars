use polars_core::utils::SuperTypeFlags;
#[cfg(feature = "is_close")]
use polars_utils::total_ord::TotalOrdWrap;

use super::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRBooleanFunction {
    Any {
        ignore_nulls: bool,
    },
    All {
        ignore_nulls: bool,
    },
    IsEmpty {
        ignore_nulls: bool,
    },
    HasNulls,
    IsNull,
    IsNotNull,
    IsFinite,
    IsInfinite,
    IsNan,
    IsNotNan,
    #[cfg(feature = "is_first_distinct")]
    IsFirstDistinct,
    #[cfg(feature = "is_last_distinct")]
    IsLastDistinct,
    #[cfg(feature = "is_unique")]
    IsUnique,
    #[cfg(feature = "is_unique")]
    IsDuplicated,
    #[cfg(feature = "is_between")]
    IsBetween {
        closed: ClosedInterval,
    },
    #[cfg(feature = "is_in")]
    IsIn {
        nulls_equal: bool,
    },
    #[cfg(feature = "is_close")]
    IsClose {
        abs_tol: TotalOrdWrap<f64>,
        rel_tol: TotalOrdWrap<f64>,
        nans_equal: bool,
    },
    IsSorted {
        descending: Option<bool>,
        nulls_last: Option<bool>,
    },
    AllHorizontal,
    AnyHorizontal,
    // Also bitwise negate
    Not,
    /// Whether the `UInt64` input is contained in a split-block bloom filter.
    ///
    /// IR-only: inserted by the distributed engine as a runtime join filter,
    /// with no DSL or Python surface. Membership is approximate — a value that
    /// was inserted is always accepted, but a value that was not may also be
    /// accepted. Nulls are always rejected.
    IsInBloomFilter {
        bitset: Arc<[u8]>,
    },
}

impl IRBooleanFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        match self {
            IRBooleanFunction::Not => {
                mapper.try_map_dtype(|dtype| {
                    match dtype {
                        DataType::Boolean => Ok(DataType::Boolean),
                        dt if dt.is_integer() => Ok(dt.clone()),
                        dt => polars_bail!(InvalidOperation: "dtype {:?} not supported in 'not' operation", dt) 
                    }
                })

            },
            _ => mapper.with_dtype(DataType::Boolean),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRBooleanFunction as B;
        match self {
            B::Any { .. } | B::All { .. } | B::IsEmpty { .. } | B::HasNulls => {
                FunctionOptions::aggregation().flag(FunctionFlags::NON_ORDER_OBSERVING)
            },
            B::IsNull | B::IsNotNull => FunctionOptions::elementwise(),
            B::IsFinite | B::IsInfinite | B::IsNan | B::IsNotNan => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::PRESERVES_NULL_FIRST_INPUT),
            #[cfg(feature = "is_first_distinct")]
            B::IsFirstDistinct => FunctionOptions::length_preserving(),
            #[cfg(feature = "is_last_distinct")]
            B::IsLastDistinct => FunctionOptions::length_preserving(),
            #[cfg(feature = "is_unique")]
            B::IsUnique => FunctionOptions::length_preserving(),
            #[cfg(feature = "is_unique")]
            B::IsDuplicated => FunctionOptions::length_preserving(),
            #[cfg(feature = "is_between")]
            B::IsBetween { .. } => FunctionOptions::elementwise()
                .with_supertyping(
                    (SuperTypeFlags::default() & !SuperTypeFlags::ALLOW_PRIMITIVE_TO_STRING).into(),
                )
                .with_flags(|f| f | FunctionFlags::PRESERVES_NULL_ALL_INPUTS),
            #[cfg(feature = "is_in")]
            B::IsIn { nulls_equal } => FunctionOptions::elementwise()
                .with_supertyping(Default::default())
                .with_flags(|f| {
                    if !*nulls_equal {
                        f | FunctionFlags::PRESERVES_NULL_FIRST_INPUT
                    } else {
                        f
                    }
                }),
            #[cfg(feature = "is_close")]
            B::IsClose { .. } => FunctionOptions::elementwise()
                .with_supertyping(
                    (SuperTypeFlags::default() & !SuperTypeFlags::ALLOW_PRIMITIVE_TO_STRING).into(),
                )
                .with_flags(|f| f | FunctionFlags::PRESERVES_NULL_ALL_INPUTS),
            B::IsSorted { .. } => FunctionOptions::aggregation(),
            // Deliberately not PRESERVES_NULL_FIRST_INPUT: a null key matches
            // nothing, so it must be rejected rather than propagated.
            B::IsInBloomFilter { .. } => FunctionOptions::elementwise(),
            B::AllHorizontal | B::AnyHorizontal => FunctionOptions::elementwise().with_flags(|f| {
                f | FunctionFlags::INPUT_WILDCARD_EXPANSION | FunctionFlags::ALLOW_EMPTY_INPUTS
            }),
            B::Not => FunctionOptions::elementwise()
                .with_flags(|f| f | FunctionFlags::PRESERVES_NULL_FIRST_INPUT),
        }
    }
}

impl Display for IRBooleanFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRBooleanFunction::*;
        let s = match self {
            All {
                ignore_nulls: false,
            } => "all",
            All { ignore_nulls: true } => "all_ignore_nulls",
            Any {
                ignore_nulls: false,
            } => "any",
            Any { ignore_nulls: true } => "any_ignore_nulls",
            IsEmpty {
                ignore_nulls: false,
            } => "is_empty",
            IsEmpty { ignore_nulls: true } => "is_empty_ignore_nulls",
            HasNulls => "has_nulls",
            IsNull => "is_null",
            IsNotNull => "is_not_null",
            IsFinite => "is_finite",
            IsInfinite => "is_infinite",
            IsNan => "is_nan",
            IsNotNan => "is_not_nan",
            #[cfg(feature = "is_first_distinct")]
            IsFirstDistinct => "is_first_distinct",
            #[cfg(feature = "is_last_distinct")]
            IsLastDistinct => "is_last_distinct",
            #[cfg(feature = "is_unique")]
            IsUnique => "is_unique",
            #[cfg(feature = "is_unique")]
            IsDuplicated => "is_duplicated",
            #[cfg(feature = "is_between")]
            IsBetween { .. } => "is_between",
            #[cfg(feature = "is_in")]
            IsIn { .. } => "is_in",
            #[cfg(feature = "is_close")]
            IsClose { .. } => "is_close",
            IsSorted { .. } => "is_sorted",
            IsInBloomFilter { .. } => "is_in_bloom_filter",
            AnyHorizontal => "any_horizontal",
            AllHorizontal => "all_horizontal",
            Not => "not",
        };
        write!(f, "{s}")
    }
}

impl From<IRBooleanFunction> for IRFunctionExpr {
    fn from(func: IRBooleanFunction) -> Self {
        IRFunctionExpr::Boolean(func)
    }
}
