use crate::internal::*;
use crate::infer::*;

#[derive(Debug, Clone, new)]
pub struct SourceState(usize);

impl OpState for SourceState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        _inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(session.inputs[&self.0].clone()))
    }
}

#[derive(Debug, Clone, new)]
pub struct Source;

impl Op for Source {
    fn name(&self) -> Cow<str> {
        "Source".into()
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatefullOp for Source {
    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(SourceState(node_id))))
    }
}

impl InferenceRulesOp for Source {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 0)?;
        check_output_arity(&outputs, 1)?;
        Ok(())
    }

    inference_op_as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        node: &InferenceNode,
        target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        use std::convert::TryFrom;
        if let Ok(fact) = TypedFact::try_from(&node.outputs[0].fact) {
            target.wire_node(&*node.name, TypedSource::new(fact), &[])
        } else {
            bail!("Output type not determined")
        }
    }
}

#[derive(Debug, Clone, new)]
pub struct TypedSource {
    fact: TypedFact,
}

impl Op for TypedSource {
    fn name(&self) -> Cow<str> {
        "TypedSource".into()
    }
    canonic!();
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatefullOp for TypedSource {
    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(SourceState(node_id))))
    }
}

impl TypedOp for TypedSource {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.fact.clone()))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let mut fact = self.fact.clone();
        change.change_shape(&mut fact.shape)?;
        Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(TypedSource::new(fact))), change)))
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        _mapping: &HashMap<OutletId, OutletId>,
        pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let pulsed_fact =
            crate::pulse::PulsedFact::from_tensor_fact_pulse(&node.outputs[0].fact, pulse)?;
        let id = target.add_source(node.name.clone(), pulsed_fact)?;
        Ok(tvec!(id))
    }

    typed_op_as_op!();
}

#[derive(Debug, Clone, new)]
pub struct PulsedSource {
    fact: PulsedFact,
}

impl Op for PulsedSource {
    fn name(&self) -> Cow<str> {
        "PulsedSource".into()
    }
    canonic!();
    not_a_typed_op!();
    op_as_pulsed_op!();
}

impl StatefullOp for PulsedSource {
    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(SourceState(node_id))))
    }
}

impl PulsedOp for PulsedSource {
    fn pulsed_output_facts(&self, _inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(self.fact.clone()))
    }

    fn to_typed(&self) -> Box<dyn TypedOp> {
        Box::new(TypedSource::new(
            TypedFact::dt_shape(self.fact.datum_type, &*self.fact.shape).unwrap(),
        ))
    }

    pulsed_op_as_op!();
}
