use tract_core::internal::*;

pub mod deser;
pub mod ser;
use crate::ast::{FragmentDecl, FragmentDef, RValue};
use crate::model::*;
use crate::ser::*;

use std::any::TypeId;

pub fn stdlib() -> Vec<FragmentDef> {
    crate::ast::parse::parse_fragments(include_str!("../../stdlib.nnef")).unwrap()
}

pub type ToTract = fn(&mut ModelBuilder, &AugmentedInvocation) -> TractResult<TVec<OutletId>>;
pub type FromTract = fn(&mut IntoAst, node: &TypedNode) -> TractResult<Arc<RValue>>;

pub struct Registry {
    id: String,
    to_tract: HashMap<String, (Option<FragmentDef>, Option<(FragmentDecl, ToTract)>)>,
    from_tract: HashMap<TypeId, (Option<FragmentDef>, Option<(FragmentDecl, FromTract)>)>,
}

impl Registry {
    pub fn new(id: impl Into<String>) -> Registry {
        Registry { id: id.into(), to_tract: Default::default(), from_tract: Default::default() }
    }

    pub fn register_primitive(&mut self, id: &str, decl: FragmentDecl, func: ToTract) {
        self.to_tract.insert(id.to_string(), (None, Some((decl, func))));
    }

    pub fn register_fragment(&mut self, def: FragmentDef) {
        self.to_tract.insert(def.decl.id.to_string(), (Some(def), None));
    }

    pub fn lookup_nnef<'a>(&'a self, id: &str) -> Option<(&FragmentDecl, ResolvedOp<'a>)> {
        self.to_tract.get(id).map(|ro| {
            if let Some(frag) = &ro.0 {
                (&frag.decl, ResolvedOp::Fragment(frag.body.as_ref().unwrap()))
            } else {
                let (decl, body) = ro.1.as_ref().unwrap();
                (decl, ResolvedOp::Primitive(body))
            }
        })
    }
}

pub fn tract_nnef() -> Registry {
    let mut reg = Registry::new("tract_nnef");
    deser::register(&mut reg);
    reg
}
