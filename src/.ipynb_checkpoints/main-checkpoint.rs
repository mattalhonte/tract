//use tract_core::ndarray;
//use tract_core::prelude::*;
use tract::*;
use rand::Rng;

fn main() -> TractResult<()> {
    // load the model
    let mut model =
        tract_tensorflow::tensorflow().model_for_path("../../saved_model/my_model.pb")?;

    // specify input type and shape
    //model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3)))?;
    

    let model = model.into_optimized()?;
    let plan = SimplePlan::new(&model)?;
    
    let mut rng = rand::thread_rng();

    let vals: Vec<u64> = (0..100).map(|_| rng.gen_range(0, 20)).collect();
    
    let result = plan.run(tvec!(vals))?;
    
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("result: {:?}", best);
    Ok(())
}
    
    