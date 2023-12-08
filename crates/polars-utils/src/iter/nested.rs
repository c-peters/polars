pub trait FromNestedIterator<A>: Sized {
    fn from_iter_nested<I: IntoIterator<Item = J>, J: IntoIterator<Item = A>>(
        iter: I,
        capacity: usize,
    ) -> Self;
}

impl<A> FromNestedIterator<A> for Vec<A> {
    fn from_iter_nested<I: IntoIterator<Item = J>, J: IntoIterator<Item = A>>(
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
        capacity: usize,
    ) -> T
    where
        Self::Item: IntoIterator,
        Self: Sized;
}

impl<K: Sized> CollectNested for K
where
    K: IntoIterator,
    K::Item: IntoIterator,
{
    fn collect_nested<T: FromNestedIterator<<Self::Item as IntoIterator>::Item>>(
        self,
        capacity: usize,
    ) -> T
    where
        Self::Item: IntoIterator,
        Self: Sized,
    {
        FromNestedIterator::from_iter_nested(self, capacity)
    }
}
