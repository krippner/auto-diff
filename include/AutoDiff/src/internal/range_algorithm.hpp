// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file range_algorithm.hpp
 * @brief Algorithms for ranges, in lieu of the C++20 ranges library.
 */

#ifndef AUTODIFF_SRC_INTERNAL_RANGE_ALGORITHM_HPP
#define AUTODIFF_SRC_INTERNAL_RANGE_ALGORITHM_HPP

#include <algorithm>
#include <iterator> // begin end
#include <utility>  // forward

namespace AutoDiff::internal {

template <typename Range, typename UnaryFunction>
auto for_each_in_range(Range&& range, UnaryFunction fct)
{
    using std::begin;
    using std::end;
    return std::for_each(begin(std::forward<Range>(range)),
        end(std::forward<Range>(range)), fct);
}

template <typename Range, typename UnaryFunction>
auto for_each_in_reversed_range(Range&& range, UnaryFunction fct)
{
    using std::rbegin;
    using std::rend;
    return std::for_each(rbegin(std::forward<Range>(range)),
        rend(std::forward<Range>(range)), fct);
}

template <typename Range, typename UnaryFunction>
auto all_of_range(Range&& range, UnaryFunction fct)
{
    using std::begin;
    using std::end;
    return std::all_of(begin(std::forward<Range>(range)),
        end(std::forward<Range>(range)), fct);
}

template <typename Range, typename UnaryFunction>
auto any_of_range(Range&& range, UnaryFunction fct)
{
    using std::begin;
    using std::end;
    return std::any_of(begin(std::forward<Range>(range)),
        end(std::forward<Range>(range)), fct);
}

template <typename Range, typename OutputIterator, typename UnaryOperation>
auto transform_range(
    Range&& range, OutputIterator firstOutput, UnaryOperation op_)
{
    using std::begin;
    using std::end;
    return std::transform(begin(std::forward<Range>(range)),
        end(std::forward<Range>(range)), firstOutput, op_);
}

template <typename Range, typename OutputIterator>
auto copy_range(Range&& range, OutputIterator firstOutput)
{
    using std::begin;
    using std::end;
    return std::copy(begin(std::forward<Range>(range)),
        end(std::forward<Range>(range)), firstOutput);
}

template <typename Range, typename OutputIterator, typename UnaryPredicate>
auto copy_range_if(
    Range&& range, OutputIterator firstOutput, UnaryPredicate pred)
{
    using std::begin;
    using std::end;
    return std::copy_if(begin(std::forward<Range>(range)),
        end(std::forward<Range>(range)), firstOutput, pred);
}

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_RANGE_ALGORITHM_HPP
