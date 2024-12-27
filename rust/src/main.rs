use std::time::Instant;

use candle_core::{DType, Device};
use candle_nn::{Module, VarBuilder};

mod dinov2;
mod imagenet;

fn main() {
    infer().unwrap();
}

fn infer() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let image = imagenet::load_image224("../image.png")?.to_device(&device)?;

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model("facebook/dinov2-base".into());
    let model_file = repo.get("model.safetensors")?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = dinov2::vit_base(vb, None)?;

    let begin = Instant::now();
    let logits = model.forward(&image.unsqueeze(0)?)?;
    let end = Instant::now();
    let elapsed = (end - begin).as_secs_f64();

    println!("Took {elapsed} seconds to evaluate {logits:?}");

    Ok(())
}
