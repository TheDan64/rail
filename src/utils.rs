use crate::{ Weight, Matrix };

pub fn array_data(array: Matrix) -> Vec<Weight> {
    let mut data: Vec<Weight> = vec![0.0; array.elements()];
    array.host(&mut data);
    data
}

pub fn map_array<T>(array: Matrix, f: T) -> Matrix
where T: Fn(&Weight) -> Weight{
    let dims = array.dims();
    let data: Vec<Weight> = array_data(array).iter().map(f).collect();
    Matrix::new(&data, dims)
}


pub fn convolve_valid(input: &Matrix, kernel: &Matrix) -> Matrix {
    use arrayfire::{ ConvMode, ConvDomain };
    let same_conv = arrayfire::convolve2(input, kernel, ConvMode::DEFAULT, ConvDomain::SPATIAL);

    same_conv
}


#[cfg(test)]
mod tests {
    use super::*;
    use arrayfire::Dim4;
    use arrayfire::print_gen;

    #[test]
    fn convolution_test() {
        let input = Matrix::new(&[
            1.0,1.0,1.0,1.0,
            1.0,1.0,1.0,1.0,
            1.0,1.0,1.0,1.0,
            1.0,1.0,1.0,1.0
        ], Dim4::new(&[4, 4, 1, 1]));

        let filter = Matrix::new(&[
            0.0,0.0,0.0,
            0.0,1.0,0.0,
            0.0,0.0,0.0
        ], Dim4::new(&[3, 3, 1, 1]));


        let output = convolve_valid(&input, &filter);
        let output_dims = output.dims().get().clone();
        let expected_dims = filter.dims().get().clone();
        assert_eq!(expected_dims, output_dims);
    }
}
