# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from typing import List, Optional, Callable, Any
from functools import partial
import torch
from contextlib import suppress
from dataclasses import dataclass
import tvm
from tilelang.utils.tensor import (
    get_tensor_supply,
    TensorSupplyType,
    torch_assert_close,
    adapt_torch2tvm,
)
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter import BaseKernelAdapter
from tilelang.profiler.bench import do_bench

import inspect
from typing import get_origin, get_args

def print_callable_info(func):
    """
    Print the names, annotations (types) and defaults of a callable's parameters,
    plus its annotated return type.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    return_annotation = sig.return_annotation

    print(f"Callable: {func.__name__}\n")

    if params:
        print("Parameters:")
        for name, param in params.items():
            # annotation or None
            ann = None if param.annotation is inspect._empty else param.annotation
            # default or None
            default = None if param.default is inspect._empty else param.default
            print(f"  • {name}")
            print(f"    – annotation: {ann}")
            print(f"    – default   : {default}")
    else:
        print("Parameters: (none)")

    # Return annotation
    ret = None if return_annotation is inspect._empty else return_annotation
    print(f"\nReturn annotation: {ret}")

@dataclass
class Profiler:
    """A profiler class for benchmarking and validating kernel implementations.
    
    Attributes:
        params: List of kernel parameters defining the input/output specifications
        result_idx: Indices indicating which parameters are output tensors
        supply_type: Type of tensor supply to use (e.g., random, zeros, etc.)
        adapter: Optional kernel adapter for interfacing with different backends
    """

    params: List[KernelParam]
    result_idx: List[int]
    supply_type: TensorSupplyType
    adapter: Optional[BaseKernelAdapter] = None

    def __post_init__(self):
        """Initialize tensor supply after dataclass initialization"""
        self.result_idx = self._legalize_result_idx(self.result_idx)
        self.supply = get_tensor_supply(self.supply_type)
        print(f"[__post_init__] self.supply={self.supply}")
        print(f"[__post_init__] Type of self.supply={type(self.supply).__name__}")
        print(f"[__post_init__] Is self.supply callable? {callable(self.supply)}")
        print_callable_info(self.supply)

    def _legalize_result_idx(self, result_idx: Optional[List[int]] = None) -> List[int]:
        params = self.params # place to initialize the params.
        print(f"[__legalize_result_idx] result_idx={result_idx}")
        # result_idx is a list of indices of the output tensors
        if result_idx is None:
            result_idx = []
        elif isinstance(result_idx, int):
            if result_idx > len(params) or result_idx < -len(params):
                raise ValueError(
                    f"result_idx should be an integer between {-len(params)} and {len(params) - 1}")
            if result_idx < 0:
                result_idx = len(params) + result_idx
            result_idx = [result_idx]
        elif not isinstance(result_idx, list):
            raise ValueError("result_idx should be a list of integers")

        return result_idx

    def with_default_adapter(self, adapter: BaseKernelAdapter) -> "Profiler":
        self.adapter = adapter
        return self

    def _get_inputs(self, with_output=False):
        ins = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                ins.append(self.supply(self.params[i]))
        return ins

    def _get_params(self, with_output=False):
        params = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                params.append(self.params[i])
        return params

    def assert_allclose(
        self,
        reference_program: Callable,
        input_tensors: Optional[List[torch.Tensor]] = None,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        max_mismatched_ratio=0.01,
    ):
        """Validates kernel output against a reference implementation.
        
        Args:
            reference_program: Reference implementation to compare against
            input_tensors: Optional pre-generated input tensors
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            max_mismatched_ratio: Maximum allowed ratio of mismatched elements
        """
        ins = self._get_inputs() if input_tensors is None else input_tensors
        ref_outs = reference_program(*ins)
        torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        torch.cuda.synchronize()
        
        print("[assert_allclose] Shapes of outputs:")
        print(f"  lib_outs: {lib_outs.shape}")    
        print(f"  ref_outs: {ref_outs.shape}")
        print("--------------------------------")
        print("[assert_allclose] Contents of outputs: ")
        print(f"  lib_outs: {lib_outs}")
        print(f"  ref_outs: {ref_outs}")
        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        elif ref_outs is None:
            ref_outs = []
        assert len(lib_outs) == len(ref_outs), "len(lib_outs) not equals to len(ref_outs) !"
        # torch.set_printoptions(edgeitems=torch.inf)
        for lhs, rhs in zip(lib_outs, ref_outs):
            # close_mask = torch.isclose(lhs, rhs, rtol=rtol, atol=atol)
            # total_elements = lhs.numel()
            # num_not_close = (~close_mask).sum().item()
            # percentage_not_close = (num_not_close / total_elements) * 100
            # print(f"{percentage_not_close:.2f}% of the elements are not close.")
            # print(f"Total elements: {total_elements}, Not close elements: {num_not_close}")
            torch_assert_close(
                lhs,
                rhs,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
            )

    def assert_consistent(self, repeat=10):
        """Checks for kernel consistency across multiple runs.
        
        Args:
            repeat: Number of times to repeat the consistency check
        """
        # Used to check no race condition inside the kernel
        ins = self._get_inputs()
        ref_outs = self.func(*ins)

        for _ in range(repeat):
            lib_outs = self.func(*ins)
            for lhs, rhs in zip(lib_outs, ref_outs):
                assert torch.allclose(lhs, rhs), [
                    "result is not consistent",
                    lhs,
                    rhs,
                ]

    def run_once(self, func: Optional[Callable] = None):
        ins = self._get_inputs()
        if not func:
            func = self.__call__
        return func(*ins)

    def determine_profiler(self, func: Optional[Callable] = None):
        """Determines which profiler backend to use based on function type.
        
        Args:
            func: Function to be profiled
            profiler: Explicitly specified profiler type or "auto" for automatic detection
        
        Returns:
            str: The determined profiler type ("torch" or "tvm")
        """
        if isinstance(func, tvm.runtime.Module):
            return "tvm"
        else:
            return "torch"

    def do_bench(
        self,
        func: Optional[Callable] = None,
        warmup: int = 25,
        rep: int = 100,
        n_warmup: int = 1,
        n_repeat: int = 1,
        input_tensors: List[torch.Tensor] = None,
    ) -> float:
        """Benchmarks the execution time of a given function.
        
        Args:
            func: Function to benchmark (uses adapter if None)
            warmup: Warmup time in milliseconds
            rep: Number of repetitions for timing
            n_warmup: Number of warmup iterations
            n_repeat: Number of timing iterations
            profiler: Which profiling backend to use
            input_tensors: Optional pre-generated input tensors
            
        Returns:
            float: Average execution time in milliseconds
        """
        profiler = self.determine_profiler(func)
        print(f"[do_bench] profiler={profiler}")
        if profiler == "torch":
            if func is None:
                assert self.adapter is not None, "benchmarking function should be provided"
                func = self.adapter
            ins = self._get_inputs() if input_tensors is None else input_tensors
            bench_func = partial(func, *ins)
            return do_bench(
                bench_func,
                warmup=warmup,
                rep=rep,
                _n_warmup=n_warmup,
                _n_repeat=n_repeat,
            )
        elif profiler == "tvm":
            assert func is not None, "func should not be None"
            assert isinstance(
                func, tvm.runtime.Module), f"func should be a TVM module, but got {type(func)}"

            ins = (self._get_inputs(with_output=True) if input_tensors is None else input_tensors)
            target = "cuda"

            with suppress(Exception):
                target = self.mod.imported_modules[0].type_key

            assert target in ["cuda", "hip"], f"Unknown target: {target}"

            device = tvm.cuda(0) if target == "cuda" else tvm.rocm(0)
            time_evaluator = self.mod.time_evaluator(
                self.mod.entry_name, device, number=rep, repeat=n_repeat)
            tvm_inputs = [adapt_torch2tvm(inp) for inp in ins]
            # Transform Latency to ms
            return time_evaluator(*tvm_inputs).mean * 1e3
        else:
            raise ValueError(f"Unknown profiler: {profiler}")

    @property
    def func(self):
        assert self.adapter is not None, "adapter should be provided"
        return self.adapter

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
