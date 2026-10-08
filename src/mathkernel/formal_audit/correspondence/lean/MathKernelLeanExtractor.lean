/-!
Bundled, data-only signature extractor for Semantic Correspondence Audit.

The Python runner replaces the marker below with exactly one validated module
import and appends one `#mathkernel_extract` command.  This helper intentionally
emits a conservative scaffold: it inspects the elaborated declaration type but
does not claim to normalize all Lean expressions into a semantic contract.
-/
import Lean
-- MATHKERNEL_TARGET_IMPORT

open Lean Elab Command Meta

namespace MathKernel.Correspondence

private def binderInfoName : BinderInfo → String
  | .default => "default"
  | .implicit => "implicit"
  | .strictImplicit => "strictImplicit"
  | .instImplicit => "instImplicit"

private def declarationKind : ConstantInfo → String
  | .axiomInfo _ => "axiom"
  | .defnInfo _ => "definition"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .quotInfo _ => "quotient"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"

private structure BinderView where
  name : String
  binderInfo : String
  type : String

private def BinderView.toJson (binder : BinderView) : Json :=
  Json.mkObj [
    ("name", toJson binder.name),
    ("binder_info", toJson binder.binderInfo),
    ("type", toJson binder.type)
  ]

private partial def inspectSignature
    (expression : Expr) (binders : Array BinderView := #[]) :
    MetaM (Array BinderView × String) := do
  match expression with
  | .forallE name domain body info =>
      let domainText := (← ppExpr domain).pretty
      withLocalDecl name info domain fun local =>
        inspectSignature (body.instantiate1 local) (binders.push {
          name := name.toString
          binderInfo := binderInfoName info
          type := domainText
        })
  | conclusion =>
      pure (binders, (← ppExpr conclusion).pretty)

private def emitSignature
    (moduleName : String) (declarationName : Name) : CommandElabM Unit := do
  let environment ← getEnv
  let some info := environment.find? declarationName
    | throwError "Declaration not found in elaborated environment: {declarationName}"
  let signature ← liftTermElabM do ppExpr info.type
  let (binders, conclusion) ← liftTermElabM do inspectSignature info.type
  let payload := Json.mkObj [
    ("schema_version", toJson "mathkernel.semantic-correspondence.lean-helper/v1"),
    ("status", toJson "scaffold"),
    ("module", toJson moduleName),
    ("declaration", toJson declarationName.toString),
    ("declaration_kind", toJson (declarationKind info)),
    ("signature", toJson signature.pretty),
    ("binders", Json.arr (binders.map BinderView.toJson)),
    ("conclusion", toJson conclusion),
    ("referenced_constants", Json.arr #[]),
    ("opaque_reasons", Json.arr #[
      toJson "Expression-tree normalization is intentionally not implemented by the version-stable scaffold.",
      toJson "Referenced constants are not claimed until a toolchain-specific traversal is independently tested."
    ])
  ]
  liftIO <| IO.println payload.compress

syntax (name := mathkernelExtract) "#mathkernel_extract " str ident : command

elab_rules : command
  | `(#mathkernel_extract $moduleName:str $declaration:ident) =>
      emitSignature moduleName.getString declaration.getId

end MathKernel.Correspondence
