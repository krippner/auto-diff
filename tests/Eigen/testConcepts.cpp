#include <AutoDiff/src/Eigen/concepts.hpp>

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

template <typename T>
concept Scalar = AutoDiff::EigenAD::Scalar<T>;

TEST_CASE("EigenAD::Scalar concept", "[Eigen]")
{
    STATIC_CHECK(Scalar<int>);
    STATIC_CHECK(Scalar<float>);
    STATIC_CHECK(Scalar<double>);
    STATIC_CHECK_FALSE(Scalar<Eigen::ArrayXd>);
    STATIC_CHECK_FALSE(Scalar<Eigen::VectorXd>);
    STATIC_CHECK_FALSE(Scalar<Eigen::RowVectorXd>);
    STATIC_CHECK_FALSE(Scalar<Eigen::MatrixXd>);
}

template <typename T>
concept MatrixBase = AutoDiff::EigenAD::MatrixBase<T>;

TEST_CASE("EigenAD::MatrixBase concept", "[Eigen]")
{
    STATIC_CHECK_FALSE(MatrixBase<int>);
    STATIC_CHECK_FALSE(MatrixBase<float>);
    STATIC_CHECK_FALSE(MatrixBase<double>);
    STATIC_CHECK_FALSE(MatrixBase<Eigen::ArrayXd>);
    STATIC_CHECK(MatrixBase<Eigen::VectorXd>);
    STATIC_CHECK(MatrixBase<Eigen::RowVectorXd>);
    STATIC_CHECK(MatrixBase<Eigen::MatrixXd>);
}

template <typename T>
concept Dense = AutoDiff::EigenAD::Dense<T>;

TEST_CASE("EigenAD::Dense concept", "[Eigen]")
{
    STATIC_CHECK_FALSE(Dense<int>);
    STATIC_CHECK_FALSE(Dense<float>);
    STATIC_CHECK_FALSE(Dense<double>);
    STATIC_CHECK(Dense<Eigen::ArrayXd>);
    STATIC_CHECK(Dense<Eigen::VectorXd>);
    STATIC_CHECK(Dense<Eigen::RowVectorXd>);
    STATIC_CHECK(Dense<Eigen::MatrixXd>);
}

template <typename T>
concept Array = AutoDiff::EigenAD::Array<T>;

TEST_CASE("EigenAD::Array concept", "[Eigen]")
{
    STATIC_CHECK_FALSE(Array<int>);
    STATIC_CHECK_FALSE(Array<float>);
    STATIC_CHECK_FALSE(Array<double>);
    STATIC_CHECK(Array<Eigen::ArrayXd>);
    STATIC_CHECK_FALSE(Array<Eigen::VectorXd>);
    STATIC_CHECK_FALSE(Array<Eigen::RowVectorXd>);
    STATIC_CHECK_FALSE(Array<Eigen::MatrixXd>);
}

template <typename T>
concept RowVector = AutoDiff::EigenAD::RowVector<T>;

TEST_CASE("EigenAD::RowVector concept", "[Eigen]")
{
    STATIC_CHECK_FALSE(RowVector<int>);
    STATIC_CHECK_FALSE(RowVector<float>);
    STATIC_CHECK_FALSE(RowVector<double>);
    STATIC_CHECK_FALSE(RowVector<Eigen::ArrayXd>);
    STATIC_CHECK_FALSE(RowVector<Eigen::VectorXd>);
    STATIC_CHECK(RowVector<Eigen::RowVectorXd>);
    STATIC_CHECK_FALSE(RowVector<Eigen::MatrixXd>);
}

template <typename T>
concept ColVector = AutoDiff::EigenAD::ColVector<T>;

TEST_CASE("EigenAD::ColVector concept", "[Eigen]")
{
    STATIC_CHECK_FALSE(ColVector<int>);
    STATIC_CHECK_FALSE(ColVector<float>);
    STATIC_CHECK_FALSE(ColVector<double>);
    STATIC_CHECK_FALSE(ColVector<Eigen::ArrayXd>);
    STATIC_CHECK(ColVector<Eigen::VectorXd>);
    STATIC_CHECK_FALSE(ColVector<Eigen::RowVectorXd>);
    STATIC_CHECK_FALSE(ColVector<Eigen::MatrixXd>);
}

template <typename T>
concept Matrix = AutoDiff::EigenAD::Matrix<T>;

TEST_CASE("EigenAD::Matrix concept", "[Eigen]")
{
    STATIC_CHECK_FALSE(Matrix<int>);
    STATIC_CHECK_FALSE(Matrix<float>);
    STATIC_CHECK_FALSE(Matrix<double>);
    STATIC_CHECK_FALSE(Matrix<Eigen::ArrayXd>);
    STATIC_CHECK_FALSE(Matrix<Eigen::VectorXd>);
    STATIC_CHECK_FALSE(Matrix<Eigen::RowVectorXd>);
    STATIC_CHECK(Matrix<Eigen::MatrixXd>);
}
