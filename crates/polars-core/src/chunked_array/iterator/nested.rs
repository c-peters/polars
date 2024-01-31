use arrow::array::{BinaryViewArray, PrimitiveArray, Utf8ViewArray};
use polars_utils::iter::nested::{CollectNested, FromNestedIterator};

use crate::datatypes::PolarsNumericType;
use crate::prelude::{BinaryChunked, ChunkedArray, StringChunked};
use crate::utils::NoNull;
impl<T> FromNestedIterator<Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn from_iter_nested<I: IntoIterator<Item = J>, J: IntoIterator<Item = Option<T::Native>>>(
        iter: I,
        capacity: usize,
    ) -> Self {
        let arr: PrimitiveArray<T::Native> = iter.collect_nested(capacity);
        ChunkedArray::with_chunk("", arr)
    }
}

impl<T> FromNestedIterator<T::Native> for NoNull<ChunkedArray<T>>
where
    T: PolarsNumericType,
{
    fn from_iter_nested<I: IntoIterator<Item = J>, J: IntoIterator<Item = T::Native>>(
        iter: I,
        capacity: usize,
    ) -> Self {
        let v: Vec<T::Native> = iter.collect_nested(capacity);
        let arr = ChunkedArray::new_vec("", v);
        NoNull::new(arr)
    }
}

impl<'a> FromNestedIterator<Option<&'a str>> for StringChunked {
    fn from_iter_nested<I: IntoIterator<Item = J>, J: IntoIterator<Item = Option<&'a str>>>(
        iter: I,
        capacity: usize,
    ) -> Self {
        let arr: Utf8ViewArray = iter.collect_nested(capacity);
        ChunkedArray::with_chunk("", arr)
    }
}

impl<'a> FromNestedIterator<Option<&'a [u8]>> for BinaryChunked {
    fn from_iter_nested<I: IntoIterator<Item = J>, J: IntoIterator<Item = Option<&'a [u8]>>>(
        iter: I,
        capacity: usize,
    ) -> Self {
        let arr: BinaryViewArray = iter.collect_nested(capacity);
        ChunkedArray::with_chunk("", arr)
    }
}
