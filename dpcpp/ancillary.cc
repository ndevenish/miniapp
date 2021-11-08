/* Ancilliary code removed from other places. The Code graveyard. */

/// Calculate the prefix sum of a 2^N sized array
template <typename T, size_t BLOCK_SIZE>
void calculate_prefix_sum_inplace(std::array<T, BLOCK_SIZE>& data) {
    constexpr size_t BLOCK_SIZE_BITS = clog2(BLOCK_SIZE);
    static_assert(is_power_of_two(BLOCK_SIZE));

    // We need to store the last element to convert to inclusive
    auto last_element = data[BLOCK_SIZE - 1];

    // Parallel prefix scan - upsweep a binary tree
    // After this, every 1,2,4,8,... node has the correct
    // sum of the two nodes below it
#pragma unroll
    for (int d = 0; d < BLOCK_SIZE_BITS; ++d) {
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += cpow(2, d + 1)) {
            data[k + cpow(2, d + 1) - 1] =
              data[k + cpow(2, d) - 1] + data[k + cpow(2, d + 1) - 1];
        }
    }

    // Parallel prefix downsweep the binary tree
    // After this, entire block has the correct prefix sum
    data[BLOCK_SIZE - 1] = 0;
#pragma unroll
    for (int d = BLOCK_SIZE_BITS - 1; d >= 0; --d) {
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += cpow(2, d + 1)) {
            // Save the left node value
            auto t = data[k + cpow(2, d) - 1];
            // Left node becomes parent
            data[k + cpow(2, d) - 1] = data[k + cpow(2, d + 1) - 1];
            // Right node becomes root + previous left value
            data[k + cpow(2, d + 1) - 1] += t;
        }
    }
// This calculated an exclusive sum. We want inclusive, so shift+add
#pragma unroll
    for (int i = 1; i < BLOCK_SIZE; ++i) {
        data[i - 1] = data[i];
    }
    data[BLOCK_SIZE - 1] = data[BLOCK_SIZE - 2] + last_element;
}
