//! Split-block bloom filter, as specified by the Parquet format.
//!
//! See <https://github.com/apache/parquet-format/blob/master/BloomFilter.md>.
//!
//! The bitset is a flat sequence of 32-byte blocks. A hash selects one block
//! and sets eight bits within it, one per 4-byte word. Because a filter is
//! just a bitset, two filters of equal length built with the same block layout
//! can be merged with a bitwise OR (see [`union_into`]).
//!
//! The `hash` values passed in are used directly: the high 32 bits pick the
//! block and the low 32 bits derive the bit mask. Any well-distributed 64-bit
//! hash is therefore a valid input.

/// Size of a single block, in bytes.
pub const BLOCK_BYTES: usize = 32;

/// magic numbers taken from https://github.com/apache/parquet-format/blob/master/BloomFilter.md
const SALT: [u32; 8] = [
    1203114875, 1150766481, 2284105051, 2729912477, 1884591559, 770785867, 2667333959, 1550580529,
];

fn hash_to_block_index(hash: u64, len: usize) -> usize {
    let number_of_blocks = len as u64 / 32;
    let low_hash = hash >> 32;
    let block_index = ((low_hash * number_of_blocks) >> 32) as u32;
    block_index as usize
}

fn new_mask(x: u32) -> [u32; 8] {
    let mut a = [0u32; 8];
    for i in 0..8 {
        let mask = x.wrapping_mul(SALT[i]);
        let mask = mask >> 27;
        let mask = 0x1 << mask;
        a[i] = mask;
    }
    a
}

/// loads a block from the bitset to the stack
#[inline]
fn load_block(bitset: &[u8]) -> [u32; 8] {
    let chunks = bitset.as_chunks::<4>().0;
    std::array::from_fn(|i| u32::from_le_bytes(chunks[i]))
}

/// assigns a block from the stack to `bitset`
#[inline]
fn store_block(block: [u32; 8], bitset: &mut [u8]) {
    let chunks = bitset.as_chunks_mut::<4>().0;
    for (i, x) in block.iter().enumerate() {
        chunks[i] = x.to_le_bytes();
    }
}

/// Returns whether the `hash` is in the set
pub fn is_in_set(bitset: &[u8], hash: u64) -> bool {
    let block_index = hash_to_block_index(hash, bitset.len());
    let key = hash as u32;

    let mask = new_mask(key);
    let slice = &bitset[block_index * 32..(block_index + 1) * 32];
    let block_mask = load_block(slice);

    for i in 0..8 {
        if mask[i] & block_mask[i] == 0 {
            return false;
        }
    }
    true
}

/// Inserts a new hash to the set
pub fn insert(bitset: &mut [u8], hash: u64) {
    let block_index = hash_to_block_index(hash, bitset.len());
    let key = hash as u32;

    let mask = new_mask(key);
    let slice = &mut bitset[block_index * 32..(block_index + 1) * 32];
    let mut block_mask = load_block(slice);

    for i in 0..8 {
        block_mask[i] |= mask[i];
    }
    store_block(block_mask, slice);
}

/// Merges `src` into `dst` by bitwise OR.
///
/// Both must have been created with [`num_bytes_for`] (or otherwise be a whole
/// number of equally-sized blocks). A merged filter accepts everything either
/// input accepted.
///
/// # Panics
/// If the two bitsets have different lengths.
pub fn union_into(dst: &mut [u8], src: &[u8]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "bloom filters must have equal length to be merged"
    );
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= *s;
    }
}

/// The bitset size in bytes needed to hold `ndv` distinct values with a false
/// positive probability of at most `fpp`.
///
/// The result is always a non-zero multiple of [`BLOCK_BYTES`], clamped to
/// `max_bytes` (which is itself rounded down to a multiple of [`BLOCK_BYTES`]).
/// Exceeding the clamp does not make the filter incorrect, only less selective.
pub fn num_bytes_for(ndv: usize, fpp: f64, max_bytes: usize) -> usize {
    assert!(fpp > 0.0 && fpp < 1.0, "fpp must be in (0, 1), got {fpp}");

    let max_blocks = (max_bytes / BLOCK_BYTES).max(1);

    if ndv == 0 {
        return BLOCK_BYTES;
    }

    // Parquet's sizing formula for a split-block filter.
    let bits = -8.0 * ndv as f64 / (1.0 - fpp.powf(1.0 / 8.0)).ln();
    let bytes = (bits / 8.0).ceil();

    // `bytes` can be non-finite or beyond usize for absurd inputs; the clamp
    // below is what keeps this in range, so saturate rather than cast blindly.
    let blocks = if bytes.is_finite() && bytes > 0.0 {
        ((bytes / BLOCK_BYTES as f64).ceil() as u64).min(max_blocks as u64) as usize
    } else {
        max_blocks
    };

    blocks.max(1) * BLOCK_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives() {
        let n = 10_000u64;
        let n_bytes = num_bytes_for(n as usize, 0.01, 1 << 20);
        let mut bitset = vec![0u8; n_bytes];

        // Values are used directly as hashes, mixed so they are well distributed.
        let hash = |i: u64| i.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(31);

        for i in 0..n {
            insert(&mut bitset, hash(i));
        }
        for i in 0..n {
            assert!(is_in_set(&bitset, hash(i)), "false negative for {i}");
        }
    }

    #[test]
    fn false_positive_rate_within_spec() {
        let n = 10_000u64;
        let fpp = 0.01;
        let n_bytes = num_bytes_for(n as usize, fpp, 1 << 20);
        let mut bitset = vec![0u8; n_bytes];

        let hash = |i: u64| i.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(31);

        for i in 0..n {
            insert(&mut bitset, hash(i));
        }

        let probes = 100_000u64;
        let false_positives = (n..n + probes)
            .filter(|&i| is_in_set(&bitset, hash(i)))
            .count();
        let observed = false_positives as f64 / probes as f64;

        // Generous headroom: this asserts the sizing is not wildly off, not
        // that the filter hits its nominal rate exactly.
        assert!(
            observed < fpp * 3.0,
            "observed fpp {observed} for target {fpp}"
        );
    }

    #[test]
    fn union_accepts_both_inputs() {
        let n_bytes = num_bytes_for(1000, 0.01, 1 << 20);
        let mut a = vec![0u8; n_bytes];
        let mut b = vec![0u8; n_bytes];

        let hash = |i: u64| i.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(31);

        for i in 0..500u64 {
            insert(&mut a, hash(i));
        }
        for i in 500..1000u64 {
            insert(&mut b, hash(i));
        }

        union_into(&mut a, &b);

        for i in 0..1000u64 {
            assert!(is_in_set(&a, hash(i)), "false negative for {i} after union");
        }
    }

    #[test]
    fn sizing_is_block_aligned_and_clamped() {
        assert_eq!(num_bytes_for(0, 0.01, 1 << 20), BLOCK_BYTES);
        assert_eq!(num_bytes_for(1_000_000_000, 0.001, 1024), 1024);
        for ndv in [1, 10, 1000, 100_000] {
            let n = num_bytes_for(ndv, 0.01, 1 << 24);
            assert_eq!(n % BLOCK_BYTES, 0);
            assert!(n > 0);
        }
    }
}
