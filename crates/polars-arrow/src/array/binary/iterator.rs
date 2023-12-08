use polars_utils::iter::nested::FromNestedIterator;

use super::{BinaryArray, MutableBinaryValuesArray};
use crate::array::{ArrayAccessor, ArrayValuesIter, MutableBinaryArray};
use crate::bitmap::utils::{BitmapIter, ZipValidity};
use crate::offset::Offset;

unsafe impl<'a, O: Offset> ArrayAccessor<'a> for BinaryArray<O> {
    type Item = &'a [u8];

    #[inline]
    unsafe fn value_unchecked(&'a self, index: usize) -> Self::Item {
        self.value_unchecked(index)
    }

    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
}

/// Iterator of values of an [`BinaryArray`].
pub type BinaryValueIter<'a, O> = ArrayValuesIter<'a, BinaryArray<O>>;

impl<'a, O: Offset> IntoIterator for &'a BinaryArray<O> {
    type Item = Option<&'a [u8]>;
    type IntoIter = ZipValidity<&'a [u8], BinaryValueIter<'a, O>, BitmapIter<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator of values of an [`MutableBinaryValuesArray`].
pub type MutableBinaryValuesIter<'a, O> = ArrayValuesIter<'a, MutableBinaryValuesArray<O>>;

impl<'a, O: Offset> IntoIterator for &'a MutableBinaryValuesArray<O> {
    type Item = &'a [u8];
    type IntoIter = MutableBinaryValuesIter<'a, O>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> FromNestedIterator<Option<&'a [u8]>> for BinaryArray<i64> {
    fn _from_iter_nested<
        I: IntoIterator<Item = J> + Clone,
        J: IntoIterator<Item = Option<&'a [u8]>>,
    >(
        iter: I,
        capacity: usize,
    ) -> Self {
        let mut arr = MutableBinaryArray::with_capacity(capacity);
        for inner_iter in iter.into_iter() {
            arr.extend(inner_iter)
        }
        arr.into()
    }
}
