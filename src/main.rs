use tract_core::ndarray;
use tract_core::prelude::*;
//use tract::*;
//use rand::{thread_rng, Rng};
use rand::*;
//use rand::distributions::Uniform;

//use rand::distributions::Standard;

fn main() -> TractResult<()> {
//fn main() {
    // load the model
    //let mut model = tract_tensorflow::tensorflow().model_for_path("./my_model.pb")?;
    //model.auto_outputs().unwrap();


    let model = tract_tensorflow::tensorflow().model_for_path("./my_model.pb")?;
    println!("Loaded model");
    //model.set_input_fact(0, InferenceFact::shape(tvec!(1, 100)))?;

    //let model = tract_tensorflow::tensorflow().model_for_path("./my_model.pb").unwrap();
        //tract_tensorflow::tensorflow().model_for_path("../../saved_model/my_model.pb")?;

    // specify input type and shape
    //model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3)))?;
    
    //model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 100, 1, 1)))?;

    //model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 100, 0, 0)))?;
    //model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(_, 100,)))?;
    //model.set_input_fact(0, InferenceFact::shape(tvec!(100)))?;
    //model.set_input_fact(0, InferenceFact::dt(f32::datum_type()))?;
    //model.auto_outputs().unwrap();

    //model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(100,)))?;
    //let rng = thread_rng();
    let mut rng = thread_rng();
    //let vals: Vec<_> = thread_rng.sample_iter(&Standard).take(100).collect();
    //let vals: Vec<_> = (0..100).map(|_| rng.gen_range::<f32>(0, 20)).collect();
    let vals: Vec<_> = (0..100).map(|_| rng.gen::<f32>()).collect();
    //let input = tensor1(&vals);
    //let input = ndarray::arr1(&vals);
    let input = ndarray::arr1(&vals).into_shape((1, 100)).unwrap();
    println!("{:?}", &vals);
    //let input = ndarray::arr2(&vals);
    //println!("2D array is {:?}", &vals);
    println!("{:?}", &input);

    //let model = model.auto_outputs().unwrap();

    //let model = model.auto_outputs()?;
    //let model = model.into_optimized()?;
    //let model = model.auto_outputs().unwrap();
    let plan = SimplePlan::new(&model).unwrap();
    //let model = model.auto_outputs().into_optimized()?;
    //let plan = SimplePlan::new(&model)?;
    
    //let mut rng = rand::thread_rng();

    //let vals: Vec<f64> = (0..100).map(|_| rng.gen_range(0, 20)).collect();
   // let result = plan.run(tvec!(input))?;
   let result = plan.run(tvec![input.into()]).unwrap();
    //let tvals: Tensor = ndarray::Array::from_shape_vec((1, 100), vals)?
    
    //let tvals = ndarray::Array4::from_shape_vec((1, 100), vals)?

    //let tvals: Tensor = ndarray::arr1::from_shape_vec((1, 100), vals)?
    
    //let result = plan.run(tvec!(tvals))?;
    
    //let result = plan.run(tvec!(tvals.into()));
    //let best = result
    //    .to_array_view::<f32>()?
    //    .iter()
    //    .cloned()
     //   .zip(1..)
     //   .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let to_show = result[0].to_array_view::<f32>()?;
    //println!("result: {:?}", result);
    println!("result: {:?}", to_show);
    Ok(())
    //result
}
    
    