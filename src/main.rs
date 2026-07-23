mod alpha;

use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;
use alpha::models::vision_transformer::VisionTransformer;

fn main() {
    let device = Default::default();
    let model = VisionTransformer::<NdArray>::new(&device, 8, 16, 3);
    let input = Tensor::<NdArray, 4>::zeros([1, 3, 16, 16], &device);
    let output = model.forward(input);
    println!("output shape: {:?}", output.shape().dims);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_transformer_smoke_test() {
        let device = Default::default();
        let model = VisionTransformer::<NdArray>::new(&device, 8, 16, 3);
        let input = Tensor::<NdArray, 4>::zeros([1, 3, 16, 16], &device);
        let output = model.forward(input);
        assert_eq!(output.shape().dims, vec![1, 8]);
    }
}
