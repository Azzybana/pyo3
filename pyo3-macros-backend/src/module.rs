//! Code generation for the function that initializes a python module and adds classes and function.

#[cfg(feature = "experimental-inspect")]
use crate::introspection::{
    attribute_introspection_code, introspection_id_const, module_introspection_code,
};
#[cfg(feature = "experimental-inspect")]
use crate::utils::expr_to_python;
use crate::{
    attributes::{
        self, kw, take_attributes, take_pyo3_options, CrateAttribute, GILUsedAttribute,
        ModuleAttribute, NameAttribute, SubmoduleAttribute,
    },
    combine_errors::CombineErrors,
    get_doc,
    pyclass::PyClassPyO3Option,
    pyfunction::{impl_wrap_pyfunction, PyFunctionOptions},
    utils::{
        has_attribute, has_attribute_with_namespace, Ctx, IdentOrStr, PyO3CratePath, PythonDoc,
    },
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::ffi::CString;
use syn::LitCStr;
use syn::{
    ext::IdentExt,
    parse::{Parse, ParseStream},
    parse_quote, parse_quote_spanned,
    punctuated::Punctuated,
    spanned::Spanned,
    token::Comma,
    Item, Meta, Path, Result,
};

#[derive(Default)]
pub struct PyModuleOptions {
    krate: Option<CrateAttribute>,
    name: Option<NameAttribute>,
    module: Option<ModuleAttribute>,
    submodule: Option<kw::submodule>,
    gil_used: Option<GILUsedAttribute>,
}

impl Parse for PyModuleOptions {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut options: PyModuleOptions = PyModuleOptions::default();

        options.add_attributes(
            Punctuated::<PyModulePyO3Option, syn::Token![,]>::parse_terminated(input)?,
        )?;

        Ok(options)
    }
}

impl PyModuleOptions {
    fn take_pyo3_options(&mut self, attrs: &mut Vec<syn::Attribute>) -> Result<()> {
        self.add_attributes(take_pyo3_options(attrs)?)
    }

    fn add_attributes(
        &mut self,
        attrs: impl IntoIterator<Item = PyModulePyO3Option>,
    ) -> Result<()> {
        macro_rules! set_option {
            ($key:ident $(, $extra:literal)?) => {
                {
                    ensure_spanned!(
                        self.$key.is_none(),
                        $key.span() => concat!("`", stringify!($key), "` may only be specified once" $(, $extra)?)
                    );
                    self.$key = Some($key);
                }
            };
        }
        attrs
            .into_iter()
            .map(|attr| {
                match attr {
                    PyModulePyO3Option::Crate(krate) => set_option!(krate),
                    PyModulePyO3Option::Name(name) => set_option!(name),
                    PyModulePyO3Option::Module(module) => set_option!(module),
                    PyModulePyO3Option::Submodule(submodule) => set_option!(
                        submodule,
                        " (it is implicitly always specified for nested modules)"
                    ),
                    PyModulePyO3Option::GILUsed(gil_used) => {
                        set_option!(gil_used);
                    }
                }

                Ok(())
            })
            .try_combine_syn_errors()?;
        Ok(())
    }
}

fn extract_use_items(
    source: &syn::UseTree,
    cfg_attrs: &[syn::Attribute],
    target_items: &mut Vec<syn::Ident>,
    target_cfg_attrs: &mut Vec<Vec<syn::Attribute>>,
) -> Result<()> {
    match source {
        syn::UseTree::Name(name) => {
            target_items.push(name.ident.clone());
            target_cfg_attrs.push(cfg_attrs.to_vec());
        }
        syn::UseTree::Path(path) => {
            extract_use_items(&path.tree, cfg_attrs, target_items, target_cfg_attrs)?;
        }
        syn::UseTree::Group(group) => {
            for tree in &group.items {
                extract_use_items(tree, cfg_attrs, target_items, target_cfg_attrs)?;
            }
        }
        syn::UseTree::Glob(glob) => {
            bail_spanned!(glob.span() => "#[pymodule] cannot import glob statements")
        }
        syn::UseTree::Rename(rename) => {
            target_items.push(rename.rename.clone());
            target_cfg_attrs.push(cfg_attrs.to_vec());
        }
    }
    Ok(())
}

struct ModuleProcessingState {
    module_items: Vec<syn::Ident>,
    module_items_cfg_attrs: Vec<Vec<syn::Attribute>>,
    introspection_chunks: Vec<TokenStream>,
    pymodule_init: Option<TokenStream>,
    module_consts: Vec<syn::Ident>,
    module_consts_cfg_attrs: Vec<Vec<syn::Attribute>>,
}

fn process_use_item(
    state: &mut ModuleProcessingState,
    item_use: &mut syn::ItemUse,
    _pyo3_path: &PyO3CratePath,
) -> Result<()> {
    let is_pymodule_export = find_and_remove_attribute(&mut item_use.attrs, "pymodule_export");
    if is_pymodule_export {
        let cfg_attrs = get_cfg_attributes(&item_use.attrs);
        extract_use_items(
            &item_use.tree,
            &cfg_attrs,
            &mut state.module_items,
            &mut state.module_items_cfg_attrs,
        )?;
    }
    Ok(())
}

fn process_fn_item(
    state: &mut ModuleProcessingState,
    item_fn: &mut syn::ItemFn,
    span: Span,
    pyo3_path: &PyO3CratePath,
) -> Result<()> {
    ensure_spanned!(
        !has_attribute(&item_fn.attrs, "pymodule_export"),
        span => "`#[pymodule_export]` may only be used on `use` or `const` statements"
    );
    let ident = &item_fn.sig.ident;
    let is_pymodule_init = find_and_remove_attribute(&mut item_fn.attrs, "pymodule_init");
    if is_pymodule_init {
        ensure_spanned!(
            !has_attribute(&item_fn.attrs, "pyfunction"),
            item_fn.span() => "`#[pyfunction]` cannot be used alongside `#[pymodule_init]`"
        );
        ensure_spanned!(state.pymodule_init.is_none(), item_fn.span() => "only one `#[pymodule_init]` may be specified");
        state.pymodule_init = Some(quote! { #ident(module)?; });
    } else if has_attribute(&item_fn.attrs, "pyfunction")
        || has_attribute_with_namespace(&item_fn.attrs, Some(pyo3_path.clone()), &["pyfunction"])
        || has_attribute_with_namespace(
            &item_fn.attrs,
            Some(pyo3_path.clone()),
            &["prelude", "pyfunction"],
        )
    {
        state.module_items.push(ident.clone());
        state
            .module_items_cfg_attrs
            .push(get_cfg_attributes(&item_fn.attrs));
    }
    Ok(())
}

fn process_struct_item(
    state: &mut ModuleProcessingState,
    item_struct: &mut syn::ItemStruct,
    span: Span,
    pyo3_path: &PyO3CratePath,
    full_name: &str,
) -> Result<()> {
    ensure_spanned!(
        !has_attribute(&item_struct.attrs, "pymodule_export"),
        span => "`#[pymodule_export]` may only be used on `use` or `const` statements"
    );
    if has_attribute(&item_struct.attrs, "pyclass")
        || has_attribute_with_namespace(&item_struct.attrs, Some(pyo3_path.clone()), &["pyclass"])
        || has_attribute_with_namespace(
            &item_struct.attrs,
            Some(pyo3_path.clone()),
            &["prelude", "pyclass"],
        )
    {
        state.module_items.push(item_struct.ident.clone());
        state
            .module_items_cfg_attrs
            .push(get_cfg_attributes(&item_struct.attrs));
        if !has_pyo3_module_declared::<PyClassPyO3Option>(
            &item_struct.attrs,
            "pyclass",
            |option| matches!(option, PyClassPyO3Option::Module(_)),
        )? {
            set_module_attribute(&mut item_struct.attrs, full_name);
        }
    }
    Ok(())
}

fn process_enum_item(
    state: &mut ModuleProcessingState,
    item_enum: &mut syn::ItemEnum,
    span: Span,
    pyo3_path: &PyO3CratePath,
    full_name: &str,
) -> Result<()> {
    ensure_spanned!(
        !has_attribute(&item_enum.attrs, "pymodule_export"),
        span => "`#[pymodule_export]` may only be used on `use` or `const` statements"
    );
    if has_attribute(&item_enum.attrs, "pyclass")
        || has_attribute_with_namespace(&item_enum.attrs, Some(pyo3_path.clone()), &["pyclass"])
        || has_attribute_with_namespace(
            &item_enum.attrs,
            Some(pyo3_path.clone()),
            &["prelude", "pyclass"],
        )
    {
        state.module_items.push(item_enum.ident.clone());
        state
            .module_items_cfg_attrs
            .push(get_cfg_attributes(&item_enum.attrs));
        if !has_pyo3_module_declared::<PyClassPyO3Option>(&item_enum.attrs, "pyclass", |option| {
            matches!(option, PyClassPyO3Option::Module(_))
        })? {
            set_module_attribute(&mut item_enum.attrs, full_name);
        }
    }
    Ok(())
}

fn process_mod_item(
    state: &mut ModuleProcessingState,
    item_mod: &mut syn::ItemMod,
    span: Span,
    pyo3_path: &PyO3CratePath,
    full_name: &str,
) -> Result<()> {
    ensure_spanned!(
        !has_attribute(&item_mod.attrs, "pymodule_export"),
        span => "`#[pymodule_export]` may only be used on `use` or `const` statements"
    );
    if has_attribute(&item_mod.attrs, "pymodule")
        || has_attribute_with_namespace(&item_mod.attrs, Some(pyo3_path.clone()), &["pymodule"])
        || has_attribute_with_namespace(
            &item_mod.attrs,
            Some(pyo3_path.clone()),
            &["prelude", "pymodule"],
        )
    {
        state.module_items.push(item_mod.ident.clone());
        state
            .module_items_cfg_attrs
            .push(get_cfg_attributes(&item_mod.attrs));
        if !has_pyo3_module_declared::<PyModulePyO3Option>(&item_mod.attrs, "pymodule", |option| {
            matches!(option, PyModulePyO3Option::Module(_))
        })? {
            set_module_attribute(&mut item_mod.attrs, full_name);
        }
        item_mod
            .attrs
            .push(parse_quote_spanned!(item_mod.mod_token.span()=> #[pyo3(submodule)]));
    }
    Ok(())
}

#[allow(unused_variables)]
fn process_const_item(
    state: &mut ModuleProcessingState,
    item_const: &mut syn::ItemConst,
    pyo3_path: &PyO3CratePath,
) {
    let is_pymodule_export = find_and_remove_attribute(&mut item_const.attrs, "pymodule_export");
    if !is_pymodule_export {
        return;
    }
    let cfg_attrs = get_cfg_attributes(&item_const.attrs);
    state.module_consts.push(item_const.ident.clone());
    state.module_consts_cfg_attrs.push(cfg_attrs.clone());
    #[cfg(feature = "experimental-inspect")]
    {
        let chunk = attribute_introspection_code(
            pyo3_path,
            None,
            item_const.ident.unraw().to_string(),
            expr_to_python(&item_const.expr),
            (*item_const.ty).clone(),
            true,
        );
        state.introspection_chunks.push(quote! {
            #(#cfg_attrs)*
            #chunk
        });
    }
}

fn ensure_no_pymodule_export(item: &syn::Item) -> Result<()> {
    let (attrs, span) = match item {
        Item::ForeignMod(i) => (&i.attrs, i.span()),
        Item::Trait(i) => (&i.attrs, i.span()),
        Item::Static(i) => (&i.attrs, i.span()),
        Item::Macro(i) => (&i.attrs, i.span()),
        Item::ExternCrate(i) => (&i.attrs, i.span()),
        Item::Impl(i) => (&i.attrs, i.span()),
        Item::TraitAlias(i) => (&i.attrs, i.span()),
        Item::Type(i) => (&i.attrs, i.span()),
        Item::Union(i) => (&i.attrs, i.span()),
        _ => return Ok(()),
    };
    ensure_spanned!(
        !has_attribute(attrs, "pymodule_export"),
        span => "`#[pymodule_export]` may only be used on `use` or `const` statements"
    );
    Ok(())
}

fn process_module_item(
    state: &mut ModuleProcessingState,
    item: &mut syn::Item,
    pyo3_path: &PyO3CratePath,
    full_name: &str,
) -> Result<()> {
    let span = item.span();
    match item {
        Item::Use(ref mut item_use) => process_use_item(state, item_use, pyo3_path),
        Item::Fn(ref mut item_fn) => process_fn_item(state, item_fn, span, pyo3_path),
        Item::Struct(ref mut item_struct) => {
            process_struct_item(state, item_struct, span, pyo3_path, full_name)
        }
        Item::Enum(ref mut item_enum) => {
            process_enum_item(state, item_enum, span, pyo3_path, full_name)
        }
        Item::Mod(ref mut item_mod) => {
            process_mod_item(state, item_mod, span, pyo3_path, full_name)
        }
        Item::Const(ref mut item_const) => {
            process_const_item(state, item_const, pyo3_path);
            Ok(())
        }
        _ => ensure_no_pymodule_export(&*item),
    }
}

/// Implement `#[pymodule]` for modules.
///
/// # Errors
///
/// Returns an error if:
/// - The module attributes are invalid.
/// - The module structure is incorrect (e.g., not an inline module).
/// - Processing of module items fails.
/// - Getting documentation fails.
pub fn pymodule_module_impl(
    module: &mut syn::ItemMod,
    mut options: PyModuleOptions,
) -> Result<TokenStream> {
    let syn::ItemMod {
        attrs,
        vis,
        unsafety: _,
        ident,
        mod_token,
        content,
        semi: _,
    } = module;
    let Some((_, items)) = content else {
        bail_spanned!(mod_token.span() => "`#[pymodule]` can only be used on inline modules")
    };
    options.take_pyo3_options(attrs)?;
    let ctx = &Ctx::new(options.krate.as_ref(), None);
    let Ctx { pyo3_path, .. } = ctx;
    let doc = get_doc(attrs, None, ctx)?;
    let name = options
        .name
        .map_or_else(|| ident.unraw(), |name| name.value.0);
    let full_name = if let Some(module) = &options.module {
        format!("{}.{}", module.value.value(), name)
    } else {
        name.to_string()
    };

    let mut state = ModuleProcessingState {
        module_items: Vec::new(),
        module_items_cfg_attrs: Vec::new(),
        introspection_chunks: Vec::new(),
        pymodule_init: None,
        module_consts: Vec::new(),
        module_consts_cfg_attrs: Vec::new(),
    };

    let _: Vec<()> = (*items)
        .iter_mut()
        .map(|item| process_module_item(&mut state, item, pyo3_path, &full_name))
        .try_combine_syn_errors()?;

    let mut add_items = TokenStream::new();
    for (item, cfg_attrs) in state.module_items.iter().zip(&state.module_items_cfg_attrs) {
        add_items.extend(quote! {
            #(#cfg_attrs)*
            #item::_PYO3_DEF.add_to_module(module)?;
        });
    }

    let mut add_consts = TokenStream::new();
    for (const_ident, cfg_attrs) in state
        .module_consts
        .iter()
        .zip(&state.module_consts_cfg_attrs)
    {
        let const_name = const_ident.unraw().to_string();
        add_consts.extend(quote! {
            #(#cfg_attrs)*
            #pyo3_path::types::PyModuleMethods::add(module, #const_name, #const_ident)?;
        });
    }

    let pymodule_init = state
        .pymodule_init
        .as_ref()
        .map_or(quote! {}, |ts| quote! { #ts });

    #[cfg(feature = "experimental-inspect")]
    let introspection = module_introspection_code(
        pyo3_path,
        &name.to_string(),
        &state.module_items,
        &state.module_items_cfg_attrs,
        state.pymodule_init.is_some(),
    );
    #[cfg(not(feature = "experimental-inspect"))]
    let introspection = quote! {};
    #[cfg(feature = "experimental-inspect")]
    let introspection_id = introspection_id_const();
    #[cfg(not(feature = "experimental-inspect"))]
    let introspection_id = quote! {};

    let gil_used = options.gil_used.is_some_and(|op| op.value.value);

    let initialization = module_initialization(
        &name,
        ctx,
        &quote! { __pyo3_pymodule },
        options.submodule.is_some(),
        gil_used,
        &doc,
    );

    let introspection_chunks = &state.introspection_chunks;

    Ok(quote!(
        #(#attrs)*
        #vis #mod_token #ident {
            #(#items)*

            #initialization
            #introspection
            #introspection_id
            #(#introspection_chunks)*

            fn __pyo3_pymodule(module: &#pyo3_path::Bound<'_, #pyo3_path::types::PyModule>) -> #pyo3_path::PyResult<()> {
                use #pyo3_path::impl_::pymodule::PyAddToModule;
                #add_items
                #add_consts
                #pymodule_init
                ::std::result::Result::Ok(())
            }
        }
    ))
}

/// Generates the function that is called by the python interpreter to initialize the native
/// module
/// Implement `#[pymodule]` for a function.
///
/// # Errors
///
/// Returns an error if:
/// - The module attributes are invalid.
/// - Processing of `#[pyfn]` attributes fails.
/// - Getting documentation fails.
pub fn pymodule_function_impl(
    function: &mut syn::ItemFn,
    mut options: PyModuleOptions,
) -> Result<TokenStream> {
    options.take_pyo3_options(&mut function.attrs)?;
    process_functions_in_module(&options, function)?;
    let ctx = &Ctx::new(options.krate.as_ref(), None);
    let Ctx { pyo3_path, .. } = ctx;
    let ident = &function.sig.ident;
    let name = options
        .name
        .map_or_else(|| ident.unraw(), |name| name.value.0);
    let vis = &function.vis;
    let doc = get_doc(&function.attrs, None, ctx)?;

    let gil_used = options.gil_used.is_some_and(|op| op.value.value);

    let initialization = module_initialization(
        &name,
        ctx,
        &quote! { ModuleExec::__pyo3_module_exec },
        false,
        gil_used,
        &doc,
    );

    #[cfg(feature = "experimental-inspect")]
    let introspection =
        module_introspection_code(pyo3_path, &name.unraw().to_string(), &[], &[], true);
    #[cfg(not(feature = "experimental-inspect"))]
    let introspection = quote! {};
    #[cfg(feature = "experimental-inspect")]
    let introspection_id = introspection_id_const();
    #[cfg(not(feature = "experimental-inspect"))]
    let introspection_id = quote! {};

    // Module function called with optional Python<'_> marker as first arg, followed by the module.
    let mut module_args = Vec::new();
    if function.sig.inputs.len() == 2 {
        module_args.push(quote!(module.py()));
    }
    module_args
        .push(quote!(::std::convert::Into::into(#pyo3_path::impl_::pymethods::BoundRef(module))));

    Ok(quote! {
        #[doc(hidden)]
        #vis mod #ident {
            #initialization
            #introspection
            #introspection_id
        }

        // Generate the definition inside an anonymous function in the same scope as the original function -
        // this avoids complications around the fact that the generated module has a different scope
        // (and `super` doesn't always refer to the outer scope, e.g. if the `#[pymodule] is
        // inside a function body)
        #[allow(unknown_lints, non_local_definitions)]
        impl #ident::ModuleExec {
            fn __pyo3_module_exec(module: &#pyo3_path::Bound<'_, #pyo3_path::types::PyModule>) -> #pyo3_path::PyResult<()> {
                #ident(#(#module_args),*)
            }
        }
    })
}

fn module_initialization(
    name: &syn::Ident,
    ctx: &Ctx,
    module_exec: &TokenStream,
    is_submodule: bool,
    gil_used: bool,
    doc: &PythonDoc,
) -> TokenStream {
    let Ctx { pyo3_path, .. } = ctx;
    let pyinit_symbol = format!("PyInit_{name}");
    let name = name.to_string();
    let pyo3_name = LitCStr::new(&CString::new(name).unwrap(), Span::call_site());

    let mut result = quote! {
        #[doc(hidden)]
        pub const __PYO3_NAME: &'static ::std::ffi::CStr = #pyo3_name;

        // This structure exists for `fn` modules declared within `fn` bodies, where due to the hidden
        // module (used for importing) the `fn` to initialize the module cannot be seen from the #module_def
        // declaration just below.
        #[doc(hidden)]
        pub(super) struct ModuleExec;

        #[doc(hidden)]
        pub static _PYO3_DEF: #pyo3_path::impl_::pymodule::ModuleDef = {
            use #pyo3_path::impl_::pymodule as impl_;

            unsafe extern "C" fn __pyo3_module_exec(module: *mut #pyo3_path::ffi::PyObject) -> ::std::os::raw::c_int {
                #pyo3_path::impl_::trampoline::module_exec(module, #module_exec)
            }

            static SLOTS: impl_::PyModuleSlots<4> = impl_::PyModuleSlotsBuilder::new()
                .with_mod_exec(__pyo3_module_exec)
                .with_gil_used(#gil_used)
                .build();

            impl_::ModuleDef::new(__PYO3_NAME, #doc, &SLOTS)
        };
    };
    if !is_submodule {
        result.extend(quote! {
            /// This autogenerated function is called by the python interpreter when importing
            /// the module.
            #[doc(hidden)]
            #[export_name = #pyinit_symbol]
            pub unsafe extern "C" fn __pyo3_init() -> *mut #pyo3_path::ffi::PyObject {
                _PYO3_DEF.init_multi_phase()
            }
        });
    }
    result
}

/// Finds and takes care of the #[pyfn(...)] in `#[pymodule]`
///
/// # Errors
///
/// Returns an error if the pyfn attribute parsing fails or if pyfunction wrapping fails.
fn process_functions_in_module(options: &PyModuleOptions, func: &mut syn::ItemFn) -> Result<()> {
    let ctx = &Ctx::new(options.krate.as_ref(), None);
    let Ctx { pyo3_path, .. } = ctx;
    let mut stmts: Vec<syn::Stmt> = Vec::new();

    for mut stmt in func.block.stmts.drain(..) {
        if let syn::Stmt::Item(Item::Fn(func)) = &mut stmt {
            if let Some((pyfn_span, pyfn_args)) = get_pyfn_attr(&mut func.attrs)? {
                let module_name = pyfn_args.modname;
                let wrapped_function = impl_wrap_pyfunction(func, pyfn_args.options)?;
                let name = &func.sig.ident;
                let statements: Vec<syn::Stmt> = syn::parse_quote_spanned! {
                    pyfn_span =>
                    #wrapped_function
                    {
                        use #pyo3_path::types::PyModuleMethods;
                        #module_name.add_function(#pyo3_path::wrap_pyfunction!(#name, #module_name.as_borrowed())?)?;
                        #[deprecated(note = "`pyfn` will be removed in a future PyO3 version, use declarative `#[pymodule]` with `mod` instead")]
                        #[allow(dead_code)]
                        const PYFN_ATTRIBUTE: () = ();
                        const _: () = PYFN_ATTRIBUTE
                    }
                };
                stmts.extend(statements);
            }
        }
        stmts.push(stmt);
    }

    func.block.stmts = stmts;
    Ok(())
}

pub struct PyFnArgs {
    modname: Path,
    options: PyFunctionOptions,
}

impl Parse for PyFnArgs {
    #[allow(clippy::needless_pass_by_value)]
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let modname = input.parse().map_err(
            |e| err_spanned!(e.span() => "expected module as first argument to #[pyfn()]"),
        )?;

        if input.is_empty() {
            return Ok(Self {
                modname,
                options: PyFunctionOptions::default(),
            });
        }

        let _: Comma = input.parse()?;

        Ok(Self {
            modname,
            options: input.parse()?,
        })
    }
}

/// Extracts the data from the #[pyfn(...)] attribute of a function
#[allow(clippy::needless_pass_by_value)]
fn get_pyfn_attr(attrs: &mut Vec<syn::Attribute>) -> syn::Result<Option<(Span, PyFnArgs)>> {
    let mut pyfn_args: Option<(Span, PyFnArgs)> = None;

    take_attributes(attrs, |attr| {
        if attr.path().is_ident("pyfn") {
            ensure_spanned!(
                pyfn_args.is_none(),
                attr.span() => "`#[pyfn] may only be specified once"
            );
            pyfn_args = Some((attr.path().span(), attr.parse_args()?));
            Ok(true)
        } else {
            Ok(false)
        }
    })?;

    if let Some((_, pyfn_args)) = &mut pyfn_args {
        pyfn_args
            .options
            .add_attributes(take_pyo3_options(attrs)?)?;
    }

    Ok(pyfn_args)
}

fn get_cfg_attributes(attrs: &[syn::Attribute]) -> Vec<syn::Attribute> {
    attrs
        .iter()
        .filter(|attr| attr.path().is_ident("cfg"))
        .cloned()
        .collect()
}

fn find_and_remove_attribute(attrs: &mut Vec<syn::Attribute>, ident: &str) -> bool {
    let mut found = false;
    attrs.retain(|attr| {
        if attr.path().is_ident(ident) {
            found = true;
            false
        } else {
            true
        }
    });
    found
}

impl PartialEq<syn::Ident> for IdentOrStr<'_> {
    fn eq(&self, other: &syn::Ident) -> bool {
        match self {
            IdentOrStr::Str(s) => other == s,
            IdentOrStr::Ident(i) => other == i,
        }
    }
}

fn set_module_attribute(attrs: &mut Vec<syn::Attribute>, module_name: &str) {
    attrs.push(parse_quote!(#[pyo3(module = #module_name)]));
}

fn has_pyo3_module_declared<T: Parse>(
    attrs: &[syn::Attribute],
    root_attribute_name: &str,
    is_module_option: impl Fn(&T) -> bool + Copy,
) -> Result<bool> {
    for attr in attrs {
        if (attr.path().is_ident("pyo3") || attr.path().is_ident(root_attribute_name))
            && matches!(attr.meta, Meta::List(_))
        {
            for option in &attr.parse_args_with(Punctuated::<T, Comma>::parse_terminated)? {
                if is_module_option(option) {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

enum PyModulePyO3Option {
    Submodule(SubmoduleAttribute),
    Crate(CrateAttribute),
    Name(NameAttribute),
    Module(ModuleAttribute),
    GILUsed(GILUsedAttribute),
}

impl Parse for PyModulePyO3Option {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(attributes::kw::name) {
            input.parse().map(PyModulePyO3Option::Name)
        } else if lookahead.peek(syn::Token![crate]) {
            input.parse().map(PyModulePyO3Option::Crate)
        } else if lookahead.peek(attributes::kw::module) {
            input.parse().map(PyModulePyO3Option::Module)
        } else if lookahead.peek(attributes::kw::submodule) {
            input.parse().map(PyModulePyO3Option::Submodule)
        } else if lookahead.peek(attributes::kw::gil_used) {
            input.parse().map(PyModulePyO3Option::GILUsed)
        } else {
            Err(lookahead.error())
        }
    }
}
