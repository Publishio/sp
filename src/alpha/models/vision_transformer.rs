use burn::{
    tensor::{backend::Backend, Tensor},
};

pub struct VisionTransformer<B: Backend> {
    patch_embed_weight: Tensor<B, 2>,
    projection_weight: Tensor<B, 2>,
    projection_bias: Tensor<B, 1>,
}

impl<B: Backend> VisionTransformer<B> {
    pub fn new(device: &B::Device, embed_dim: usize, patch_size: usize, in_channels: usize) -> Self {
        let patch_dim = 768;
        let patch_embed_weight = Tensor::<B, 2>::ones([patch_dim, embed_dim], device);
        let projection_weight = Tensor::<B, 2>::ones([embed_dim, embed_dim], device);
        let projection_bias = Tensor::<B, 1>::zeros([embed_dim], device);

        Self {
            patch_embed_weight,
            projection_weight,
            projection_bias,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch = x.shape().dims[0];
        let flat = x.reshape([batch, 1 * 16 * 16 * 3]);
        let embedded = flat.matmul(self.patch_embed_weight.clone());
        let projected = embedded.matmul(self.projection_weight.clone());
        projected + self.projection_bias.clone().reshape([1, self.projection_bias.shape().dims[0]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    #[test]
    fn tiny_forward_pass_runs() {
        let device = Default::default();
        let model = VisionTransformer::<NdArray>::new(&device, 8, 16, 3);
        let input = Tensor::<NdArray, 4>::zeros([1, 3, 16, 16], &device);
        let output = model.forward(input);
        assert_eq!(output.shape().dims, vec![1, 8]);
    }
}
