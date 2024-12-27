// This code is not available from the library, so copy it here
// https://github.com/huggingface/candle/blob/main/candle-examples/src/imagenet.rs

use candle_core::{Device, Result, Tensor};

pub const IMAGENET_MEAN: [f32; 3] = [0.485f32, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229f32, 0.224, 0.225];

/// Loads an image from disk using the image crate at the requested resolution,
/// using the given std and mean parameters.
/// This returns a tensor with shape (3, res, res). imagenet normalization is applied.
pub fn load_image_with_std_mean<P: AsRef<std::path::Path>>(
    p: P,
    res: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Tensor> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(
            res as u32,
            res as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (res, res, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(mean, &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(std, &Device::Cpu)?.reshape((3, 1, 1))?;
    (data.to_dtype(candle_core::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}

/// Loads an image from disk using the image crate at the requested resolution.
/// This returns a tensor with shape (3, res, res). imagenet normalization is applied.
pub fn load_image<P: AsRef<std::path::Path>>(p: P, res: usize) -> Result<Tensor> {
    load_image_with_std_mean(p, res, &IMAGENET_MEAN, &IMAGENET_STD)
}

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 224, 224). imagenet normalization is applied.
pub fn load_image224<P: AsRef<std::path::Path>>(p: P) -> Result<Tensor> {
    load_image(p, 224)
}

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 518, 518). imagenet normalization is applied.
/// The model dinov2 reg4 analyzes images with dimensions 3x518x518 (resulting in 37x37 transformer tokens).
pub fn load_image518<P: AsRef<std::path::Path>>(p: P) -> Result<Tensor> {
    load_image(p, 518)
}
