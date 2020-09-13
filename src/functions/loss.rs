use crate::Matrix;
pub type LossFuntion = dyn Fn(&Matrix, &Matrix) -> f64;
pub type DLossFuntion = dyn Fn(&Matrix, &Matrix) -> Matrix;
pub type LossFuntionGenerator = dyn Fn() -> (Box<LossFuntion>, Box<DLossFuntion>);

pub fn mean_squared_error() -> (Box<LossFuntion>, Box<DLossFuntion>){
    let mse = |output: &Matrix, expected: &Matrix| {
        let diff = arrayfire::sub(expected, output, true);
        let squared = &diff * &diff;
        let mean = arrayfire::mean(&squared, 0);
        let result = mean * 0.5f64;
        use crate::utils::array_data;
        return array_data(result).iter().sum();
    };

    let d_mse = |output: &Matrix, expected: &Matrix| {
        let diff = arrayfire::sub(expected, output, true);
        let sum = arrayfire::sum(&diff, 0);

        sum
    };

    (Box::new(mse), Box::new(d_mse))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayfire::Dim4;
    use crate::utils::array_data;

    #[test]
    fn mse_test() {
        let input = Matrix::new(&[1.0, 0.0, 0.0], Dim4::new(&[3, 1, 1, 1]));
        let expected = Matrix::new(&[-0.5, 0.0, 0.0], Dim4::new(&[3, 1, 1, 1]));
        let (mse, _) = mean_squared_error();

        let output = mse(&input, &expected);
        assert_eq!(output, 0.375);

        let input = Matrix::new(&[1.0, 1.0, 1.0], Dim4::new(&[3, 1, 1, 1]));
        let expected = Matrix::new(&[1.0, 1.0, 1.0], Dim4::new(&[3, 1, 1, 1]));

        let output = mse(&input, &expected);
        assert_eq!(output, 0.0);

        let input = Matrix::new(&[0.0, 0.0, 0.0], Dim4::new(&[3, 1, 1, 1]));
        let expected = Matrix::new(&[1.0, 1.0, 1.0], Dim4::new(&[3, 1, 1, 1]));

        let output = mse(&input, &expected);
        assert_eq!(output, 0.5);
    }

    #[test]
    fn d_mse_test() {
        let output = Matrix::new(&[-0.5, 0.0, 0.0], Dim4::new(&[3, 1, 1, 1]));
        let expected = Matrix::new(&[1.0, 0.0, 0.0], Dim4::new(&[3, 1, 1, 1]));
        let (_, d_mse) = mean_squared_error();

        let output = d_mse(&output, &expected);
        let data = array_data(output);
        assert_eq!(data, vec![1.5]);

        let output = Matrix::new(&[1.0, 1.0, 1.0], Dim4::new(&[3, 1, 1, 1]));
        let expected = Matrix::new(&[1.0, 1.0, 1.0], Dim4::new(&[3, 1, 1, 1]));

        let output = d_mse(&output, &expected);
        let data = array_data(output);
        assert_eq!(data, vec![0.0]);

        let output = Matrix::new(&[0.0, 0.0, 0.0], Dim4::new(&[3, 1, 1, 1]));
        let expected = Matrix::new(&[1.0, 1.0, 1.0], Dim4::new(&[3, 1, 1, 1]));

        let output = d_mse(&output, &expected);
        let data = array_data(output);
        assert_eq!(data, vec![3.0]);
    }
}
