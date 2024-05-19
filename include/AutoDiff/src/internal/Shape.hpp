// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_INTERNAL_SHAPE_HPP
#define AUTODIFF_SRC_INTERNAL_SHAPE_HPP

#include <algorithm> // copy
#include <array>
#include <cassert>
#include <cstdint> // uint8_t
#include <cstdlib> // size_t
#include <initializer_list>

namespace AutoDiff::internal {

/**
 * @class StaticVector
 * @brief A static array with variable size.
 *
 * @tparam T           the value type of the array
 * @tparam N           the size of the static array
 */
template <typename T, std::size_t N>
class StaticVector {
public:
    StaticVector() = default;

    StaticVector(std::initializer_list<T> list)
        : mSize{list.size()}
    {
        assert(mSize <= N);
        std::copy(list.begin(), list.end(), mStorage.begin());
    }

    ~StaticVector() = default;

    StaticVector(StaticVector const&)                        = default;
    StaticVector(StaticVector&&) noexcept                    = default;
    auto operator=(StaticVector const&) -> StaticVector&     = default;
    auto operator=(StaticVector&&) noexcept -> StaticVector& = default;

    /**
     * @brief The current size of the array.
     */
    [[nodiscard]] auto size() const -> std::size_t { return mSize; }

    /**
     * @brief Returns the value at the given index.
     */
    [[nodiscard]] auto operator[](std::size_t index) const -> T
    {
        return mStorage.at(index);
    }

    [[nodiscard]] friend auto operator==(
        StaticVector const& left, StaticVector const& right) -> bool
    {
        return left.mStorage == right.mStorage;
    }

    [[nodiscard]] friend auto operator!=(
        StaticVector const& left, StaticVector const& right) -> bool
    {
        return left.mStorage != right.mStorage;
    }

private:
    std::array<T, N> mStorage{};
    std::size_t mSize{0};
};

/**
 * @brief Object shape, e.g., number of rows, columns.
 *
 * At most 8 dimensions possible.
 */
using Shape = StaticVector<std::size_t, 8>;

/**
 * @brief Characteristics of a map between objects with shape.
 */
struct MapDescription {
    enum State : std::uint8_t { evaluated, zero, identity };
    State state{evaluated};
    Shape domainShape;
    Shape codomainShape;
};

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_SHAPE_HPP
