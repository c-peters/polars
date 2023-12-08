pub trait FromNestedIterator<A>: Sized {
    fn from_iter_nested<I: IntoIterator<Item = J> + Clone, J: IntoIterator<Item = A>>(
        iter: I,
        length: Option<usize>,
    ) -> Self {
        let capacity = <Self as FromNestedIterator<A>>::_get_capacity(&iter, length);
        FromNestedIterator::_from_iter_nested(iter, capacity)
    }

    fn _get_capacity<I: IntoIterator<Item = J> + Clone, J: IntoIterator<Item = A>>(
        iter: &I,
        length: Option<usize>,
    ) -> usize {
        length.unwrap_or_else(|| {
            iter.clone()
                .into_iter()
                .map(|inner_iter| inner_iter.into_iter().size_hint().0)
                .sum()
        })
    }

    fn _from_iter_nested<I: IntoIterator<Item = J> + Clone, J: IntoIterator<Item = A>>(
        iter: I,
        capacity: usize,
    ) -> Self;
}

impl<A> FromNestedIterator<A> for Vec<A> {
    fn _from_iter_nested<I: IntoIterator<Item = J> + Clone, J: IntoIterator<Item = A>>(
        iter: I,
        capacity: usize,
    ) -> Self {
        let mut out = Vec::with_capacity(capacity);
        let iters = iter.into_iter();
        for iter in iters.into_iter() {
            for v in iter {
                out.push(v);
            }
        }
        out
    }
}

pub trait CollectNested: IntoIterator {
    fn collect_nested<T: FromNestedIterator<<Self::Item as IntoIterator>::Item>>(
        self,
        length: Option<usize>,
    ) -> T
    where
        Self::Item: IntoIterator,
        Self: Sized + Clone;
}

impl<K: Sized> CollectNested for K
where
    K: IntoIterator + Clone,
    K::Item: IntoIterator,
{
    fn collect_nested<T: FromNestedIterator<<Self::Item as IntoIterator>::Item>>(
        self,
        length: Option<usize>,
    ) -> T
    where
        Self::Item: IntoIterator,
        Self: Sized + Clone,
    {
        FromNestedIterator::from_iter_nested(self, length)
    }
}
