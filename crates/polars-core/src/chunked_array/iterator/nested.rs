use arrow::array::{BinaryArray, PrimitiveArray, Utf8Array};
use polars_utils::iter::nested::{CollectNested, FromNestedIterator};

use crate::datatypes::PolarsNumericType;
use crate::prelude::{BinaryChunked, ChunkedArray, Utf8Chunked};
use crate::utils::NoNull;

impl<T> FromNestedIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn _from_iter_nested<
        I: IntoIterator<Item = J> + Clone,
        J: IntoIterator<Item = Option<T::Native>>,
    >(
        iter: I,
        capacity: usize,
    ) -> Self {
        let arr: PrimitiveArray<T::Native> = iter.collect_nested(Some(capacity));
        ChunkedArray::with_chunk("", arr)
    }
}

impl<T> FromNestedIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    fn _from_iter_nested<I: IntoIterator<Item = J> + Clone, J: IntoIterator<Item = T::Native>>(
        iter: I,
        capacity: usize,
    ) -> Self {
        let v: Vec<T::Native> = iter.collect_nested(Some(capacity));
        let arr = ChunkedArray::new_vec("", v);
        NoNull::new(arr)
    }
}

impl<'a> FromNestedIterator<Option<&'a str>> for Utf8Chunked {
    fn _from_iter_nested<
        I: IntoIterator<Item = J> + Clone,
        J: IntoIterator<Item = Option<&'a str>>,
    >(
        iter: I,
        capacity: usize,
    ) -> Self {
        let arr: Utf8Array<i64> = iter.collect_nested(Some(capacity));
        ChunkedArray::with_chunk("", arr)
    }
}

impl<'a> FromNestedIterator<Option<&'a [u8]>> for BinaryChunked {
    fn _from_iter_nested<
        I: IntoIterator<Item = J> + Clone,
        J: IntoIterator<Item = Option<&'a [u8]>>,
    >(
        iter: I,
        capacity: usize,
    ) -> Self {
        let arr: BinaryArray<i64> = iter.collect_nested(Some(capacity));
        ChunkedArray::with_chunk("", arr)
    }
}
