from __future__ import annotations, division
import ast
import hashlib
import inspect
import threading
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Generic, Iterable, Optional, TypeVar, overload

from triton.backends import BaseBackend
from types import ModuleType
from .. import knobs
from .driver import driver
from . import _async_compile
from .._utils import find_paths_if, get_iterable_path, type_canonicalisation_dict, is_namedtuple
from .cache import get_cache_key
from triton._C.libtriton import get_cache_invalidating_env_vars, native_specialize_impl, ir

TRITON_MODULE = "triton.language"

T = TypeVar("T")

# -----------------------------------------------------------------------------
# JITFunction
# -----------------------------------------------------------------------------


def _normalize_ty(ty) -> str:
    import triton.language.core as core
    if isinstance(ty, str):
        ty = ty.strip()
        if ty.startswith("const "):
            ty = ty.removeprefix("const")
            ty = _normalize_ty(ty)
            assert ty.startswith("*")
            return "*k" + ty[1:]
        if ty.endswith("*"):
            return "*" + _normalize_ty(ty[:-1])
        if ty.startswith("*"):
            return "*" + _normalize_ty(ty[1:])
        if ty.startswith("tl."):
            return _normalize_ty(ty.removeprefix("tl."))
    elif isinstance(ty, core.pointer_type):
        return f"*{_normalize_ty(ty.element_ty)}"
    elif isinstance(ty, core.dtype):
        ty = ty.name
    elif isinstance(ty, type):
        ty = ty.__name__
    else:
        ty = str(ty)
    return type_canonicalisation_dict.get(ty.replace("_t", ""), ty)


class KernelParam:
    """Represents a parameter (name plus metadata) to a @jit'ed function."""

    def __init__(self, num: int, param: inspect.Parameter, do_not_specialize: bool,
                 do_not_specialize_on_alignment: bool):
        self.num = num
        self._param = param
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment

    @cached_property
    def name(self):
        return self._param.name

    @cached_property
    def annotation(self) -> str:
        if not self._param.annotation or self._param.annotation == inspect.Parameter.empty:
            return ""
        return _normalize_ty(self._param.annotation)

    @cached_property
    def annotation_type(self) -> str:
        a = self.annotation
        if a.startswith("*k"):
            a = a[2:]
        elif a.startswith("*"):
            a = a[1:]
        if a in set(type_canonicalisation_dict.values()):
            return self.annotation
        return ""

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @cached_property
    def is_const(self):
        if self.is_constexpr:
            return False
        return "const" in self.annotation or self.annotation.startswith("*k")

    @property
    def default(self):
        return self._param.default

    @property
    def has_default(self):
        return self._param.default != inspect.Parameter.empty


def mangle_type(arg, specialize=False):
    is_const = False
    align = True
    return native_specialize_impl(BaseBackend, arg, is_const, specialize, align)[0]


class KernelInterface(Generic[T]):
    run: T

    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)

    def run(self, *args, grid, warmup, **kwargs):
        raise NotImplementedError("run not implemented")

    def __getitem__(self, grid) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
        # return cast(T, functools.partial(cast(Callable, self.run), grid=grid))


def serialize_specialization_data(name, signature, constants, attrs, options, key, target):
    constants = {
        key: str(value) if value.__class__.__name__ == "dtype" else {"constexpr": value.value}
        if value.__class__.__name__ == "constexpr" else {"jit_function": f"{value.module}:{value.fn.__qualname__}"}
        if value.__class__.__name__ == "JITFunction" else value
        for key, value in constants.items()
    }

    import json
    obj = {
        'name': name, 'signature': signature, 'constant_keys': [list(x) for x in constants.keys()], 'constant_vals':
        list(constants.values()), 'attrs_keys': [list(x) for x in attrs.keys()], 'attrs_vals': list(attrs.values()),
        'options': options.__dict__, 'key': key, 'target': target.__dict__
    }
    serialized_obj = json.dumps(obj)
    return serialized_obj


def create_function_from_signature(sig, kparams, backend):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """
    assert len(sig.parameters) == len(kparams)
    # Create the function argument list and the dict entries for the return statement
    specialization = []
    # signature
    for name, kp in zip(sig.parameters.keys(), kparams):
        if kp.is_constexpr:
            specialization.append(f'("constexpr", {name})')
        else:
            is_const = 'True' if kp.is_const else 'False'
            specialize = 'False' if kp.do_not_specialize else 'True'
            align = 'False' if kp.do_not_specialize_on_alignment else 'True'
            ret = f"specialize_impl(backend, {name}, {is_const}, {specialize}, {align})"
            if kp.annotation_type:
                if isinstance(kp.annotation_type, str):
                    if kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]:
                        # we do not specialize non-constexpr floats and bools:
                        specialize = False
                if specialize:
                    specialization.append(f'("{kp.annotation_type}",) + {ret}[1:]')
                else:
                    # skip runtime specialization:
                    specialization.append(f'("{kp.annotation_type}", None)')
            else:
                specialization.append(f"{ret}")

    # compute argument string for a given parameter
    arg = lambda x: x[0] if x[1].default is inspect.Parameter.empty else f"{x[0]}=default_{x[0]}"
    func_body = f"""
def dynamic_func({", ".join(list(map(arg, sig.parameters.items())) + ["**options"])}):
    params = {{{', '.join([f"'{name}': {name}" for name in sig.parameters.keys()])}}}
    specialization = [{','.join(specialization)}]
    return params, specialization, options
"""

    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    specialize_impl = native_specialize_impl
    func_namespace["specialize_impl"] = specialize_impl
    func_namespace["backend"] = backend
    func_namespace["JITCallable"] = JITCallable

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace['dynamic_func']


def get_full_name(fn):
    return f"{fn.__module__}.{fn.__qualname__}"


class JITCallable:

    def __init__(self, fn):
        self.fn = fn
        self.signature = inspect.signature(fn)
        try:
            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        except OSError as e:
            raise ValueError("@jit functions should be defined in a Python file") from e
        self._fn_name = get_full_name(fn)
        self._hash_lock = threading.RLock()

        # function source code (without decorators)
        src = textwrap.dedent("".join(self.raw_src))
        src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
        self._src = src
        self.hash = None

        # reuse docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    def get_capture_scope(self):
        fn = self.fn
        if fn.__closure__ is None:
            return self.__globals__
        nonlocals = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
        return self.__globals__ | nonlocals

    @property
    def cache_key(self) -> str:
        # Triton Nano: Simplified hash using just source code
        with self._hash_lock:
            if self.hash is not None:
                return self.hash
            self.hash = hashlib.sha256((self._src + str(self.starting_line_number)).encode("utf-8")).hexdigest()
        return self.hash

    def __hash__(self):
        return hash(self.cache_key)

    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Our unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self._src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    @property
    def type(self):
        from triton.language.core import constexpr_type
        return constexpr_type(self)

    def _flatten_ir(self, handles: list[ir.value]) -> None:
        pass

    def _unsafe_update_src(self, new_src):
        """
        The only method allowed to modify src.
        Bypasses the __setattr__ restriction by calling super().__setattr__ directly.

        Note that it is the callers responsibility to make sure any triton functions that call this function have the `.hash` value reset to None.
        """
        self.hash = None
        self._src = new_src

    def _set_src(self):
        raise AttributeError("Cannot set attribute 'src' directly. "
                             "Use '_unsafe_update_src()' and manually clear `.hash` of all callers"
                             "instead.")

    def _get_src(self):
        return self._src

    src = property(fget=_get_src, fset=_set_src)


_triton_jit_function_registry = {}


@dataclass
class JitFunctionInfo:
    module: ModuleType
    name: str
    jit_function: JITFunction


def compute_cache_key(kernel_key_cache, specialization, options):
    key = (tuple(specialization), str(options))
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key

    # Replace JITCallable objects with their hash, so the cache key will change if the src is updated
    def replace_callables(obj):
        if isinstance(obj, list):
            return [replace_callables(arg) for arg in obj]
        elif is_namedtuple(obj):
            results = [replace_callables(arg) for arg in obj]
            return obj.__class__(*results)
        elif isinstance(obj, tuple):
            return tuple(replace_callables(arg) for arg in obj)
        elif isinstance(obj, JITCallable):
            return obj.cache_key
        return obj

    cache_key = str(replace_callables(specialization)) + str(options)
    kernel_key_cache[key] = cache_key
    return cache_key


def convert_to_tuple_if_list(item):
    # If the incoming item is a list, recursively iterate through it to convert all lists therein into tuples
    if not isinstance(item, list):
        return item

    # The value must be a list at this point
    for i, nested_value in enumerate(item):
        item[i] = convert_to_tuple_if_list(nested_value)

    return tuple(item)


class JITFunction(JITCallable, KernelInterface[T]):

    def is_gluon(self):
        return False

    def _call_hook(
        self,
        hook,
        key,
        signature,
        target,
        device,
        constants,
        options,
        configs,
        is_warmup,
    ) -> bool | None:
        if not hook:
            return None

        name = self.fn.__qualname__
        module = self.fn.__module__
        arg_reprs = ", ".join([f"{param.name}: {ty}" for param, ty in zip(self.params, key[1])])
        repr = f"{name}[num_warps={options.num_warps}, num_ctas={options.num_ctas}, num_stages={options.num_stages}, enable_fp_fusion={options.enable_fp_fusion}, launch_cooperative_grid={options.launch_cooperative_grid}]({arg_reprs})"
        full_name = get_full_name(self.fn)

        specialization_data = serialize_specialization_data(full_name, signature, constants, configs[0], options, key,
                                                            target)

        kwargs = {
            'signature': signature,
            'device': device,
            'constants': constants,
            'num_warps': options.num_warps,
            'num_ctas': options.num_ctas,
            'num_stages': options.num_stages,
            'enable_fp_fusion': options.enable_fp_fusion,
            'launch_cooperative_grid': options.launch_cooperative_grid,
            'extern_libs': options.extern_libs,
            'configs': configs,
            'specialization_data': specialization_data,
            'is_warmup': is_warmup,
        }

        return hook(
            key=key,
            repr=repr,
            fn=JitFunctionInfo(module, name, self),
            compile={"key": key, **kwargs},
            is_manual_warmup=is_warmup,
            already_compiled=False,
        )

    def add_pre_run_hook(self, hook):
        '''
        Add a hook that will be executed prior to the execution of run
        function with args and kwargs passed into the kernel
        '''
        assert callable(hook)
        self.pre_run_hooks.append(hook)

    def create_binder(self):
        """
        Precompute as much as possible.
        """
        from ..compiler import CompiledKernel, compile, ASTSource, make_backend
        target = driver.active.get_current_target()
        backend = make_backend(target)
        self.CompiledKernel = CompiledKernel
        self.compile = compile
        self.ASTSource = ASTSource
        binder = create_function_from_signature(self.signature, self.params, backend)
        return {}, {}, target, backend, binder

    def _pack_args(self, backend, kwargs, bound_args, specialization, options):
        # options
        options = backend.parse_options(kwargs)
        # signature
        sigkeys = [x.name for x in self.params]
        sigvals = [x[0] for x in specialization]
        signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
        # check arguments
        assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
        assert "device" not in kwargs, "device option is deprecated; current device will be used"
        assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
        for k in kwargs:
            if k not in options.__dict__ and k not in sigkeys:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        # constexprs
        constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
        # attributes
        attrvals = ['' if x[0] == 'constexpr' else x[1] for x in specialization]
        attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
        attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

        return options, signature, constexprs, attrs

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
        kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        # add a cache field to the kernel specializations for kernel specific
        # pass pipelines
        if knobs.runtime.add_stages_inspection_hook is not None:
            inspect_stages_key, inspect_stages_hash = knobs.runtime.add_stages_inspection_hook()
            specialization.append(f'("custom_pipeline", {inspect_stages_hash})')

        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            options, signature, constexprs, attrs = self._pack_args(backend, kwargs, bound_args, specialization,
                                                                    options)

            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
            if kernel is None:
                return None

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel

    def repr(self, _):
        return self._fn_name if self._repr is None else self._repr(_)

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        do_not_specialize = do_not_specialize if do_not_specialize else []
        do_not_specialize_on_alignment = do_not_specialize_on_alignment if do_not_specialize_on_alignment else []

        super().__init__(fn)
        self.module = fn.__module__
        self.version = version
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment
        self._repr = repr
        self.launch_metadata = launch_metadata
        # Register for simple deserialization of JITFunction constants
        _triton_jit_function_registry[f"{self.module}:{self.fn.__qualname__}"] = self

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = i in do_not_specialize or param.name in do_not_specialize
            dns_oa = i in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
            self.params.append(KernelParam(i, param, dns, dns_oa))

        # cache of just-in-time compiled kernels
        self.device_caches = defaultdict(self.create_binder)

        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel = None
        self.debug = debug
        self.noinline = noinline

        # TODO(jlebar): Remove uses of these fields outside this file, then
        # remove the fields here.
        self.arg_names = [p.name for p in self.params]
        self.constexprs = [p.num for p in self.params if p.is_constexpr]

        # Hooks that will be called prior to executing "run"
        self.pre_run_hooks = []

    def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
        kernel_cache, _, target, backend, _ = self.device_caches[device]

        if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, target, device, constexprs, options, [attrs],
                           warmup):
            return None
        src = self.ASTSource(self, signature, constexprs, attrs)

        async_mode = _async_compile.active_mode.get()
        if async_mode is not None:

            env_vars = get_cache_invalidating_env_vars()
            cache_key = get_cache_key(src, backend, options, env_vars)

            def async_compile():
                return self.compile(src, target=target, options=options.__dict__, _env_vars=env_vars)

            def finalize_compile(kernel):
                kernel_cache[key] = kernel
                self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, target, device, constexprs,
                                options, [attrs], warmup)

            kernel = async_mode.submit(cache_key, async_compile, finalize_compile)
            kernel_cache[key] = kernel
        else:
            kernel = self.compile(src, target=target, options=options.__dict__)
            kernel_cache[key] = kernel
            self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, target, device, constexprs, options,
                            [attrs], warmup)
        return kernel

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")

    def __repr__(self):
        return f"JITFunction({self.module}:{self.fn.__qualname__})"


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------


@overload
def jit(fn: T) -> JITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], JITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> KernelInterface[T]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        assert callable(fn)
        return JITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
            do_not_specialize_on_alignment=do_not_specialize_on_alignment,
            debug=debug,
            noinline=noinline,
            repr=repr,
            launch_metadata=launch_metadata,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator


# -----------------------------------------------------------------------------
# Utilities for mocking tensors
# -----------------------------------------------------------------------------


class MockTensor:
    """
    Can be used in place of real tensors when calling:
        kernel.warmup(MockTensor(torch.float32), ...)
    """

    @staticmethod
    def wrap_dtype(arg):
        if arg.__class__.__name__ == "dtype" and arg.__module__ == "torch":
            return MockTensor(arg)
        return arg

    def __init__(self, dtype, shape=None):
        if shape is None:
            shape = [1]
        self.dtype = dtype
        self.shape = shape

    def stride(self):
        strides = [1]
        for size in self.shape[1:]:
            strides.append(strides[-1] * size)
        return tuple(reversed(strides))

    @staticmethod
    def data_ptr():
        return 0  # optimistically assumes multiple of 16

    @staticmethod
    def ptr_range():
        return 0  # optimistically assumes 32 bit pointer range



def get_jit_fn_file_line(fn):
    base_fn = fn
    while not isinstance(base_fn, JITCallable):
        base_fn = base_fn.fn
    file_name = base_fn.fn.__code__.co_filename
    begin_line = base_fn.starting_line_number
    # Match the following pattern:
    # @triton.autotune(...) <- foo.__code__.co_firstlineno
    # @triton.heuristics(...)
    # @triton.jit
    # def foo(...): <- this line is the first line
    for idx, line in enumerate(base_fn.raw_src):
        if line.strip().startswith("def "):
            begin_line += idx
            break
    return file_name, begin_line


class BoundConstexprFunction(JITCallable):

    def __init__(self, instance, fn):
        self.__self__ = instance
        self.__func__ = fn

    @property
    def cache_key(self):
        return self.__func__.cache_key

    def __call__(self, *args, **kwargs):
        return self.__func__(self.__self__, *args, **kwargs)


class ConstexprFunction(JITCallable):

    def __init__(self, fn):
        super().__init__(fn)

    def __get__(self, obj, objclass):
        # Create a bound function to support constexpr_function methods
        if obj is not None:
            return BoundConstexprFunction(obj, self)
        return self

    def __call__(self, *args, _semantic=None, **kwargs):
        from triton.language.core import _unwrap_if_constexpr, constexpr
        # de-constexpr arguments and discard the _semantic keyword argument:
        args = [_unwrap_if_constexpr(x) for x in args]
        kwargs = {k: _unwrap_if_constexpr(v) for (k, v) in kwargs.items()}

        # call the raw Python function f:
        res = self.fn(*args, **kwargs)

        if _semantic is None:
            # Not called by triton code generator, e.g. in host code, another constexpr function, or even an aggreate's __init__ function
            return res

        # convert result back to a Triton constexpr:
        if knobs.runtime.interpret:
            return res  # No constexpr in interpreter
        return constexpr(res)


def constexpr_function(fn):
    """
    Wraps an arbitrary Python function so that it can be called at
    compile-time on constexpr arguments in a Triton function and
    returns a constexpr result.
    """
    return ConstexprFunction(fn)
