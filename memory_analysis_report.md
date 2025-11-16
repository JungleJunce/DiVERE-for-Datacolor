# Memory Issue Analysis Report

## Executive Summary

**Status**: ✅ **The issues described in `memory.md` are SUBSTANTIALLY VALID**

The heap high-water mark problem is real and stems from the preview pipeline's array allocation pattern. However, some assumptions in `memory.md` about cache sizes are outdated - caches are already well-limited. The core issue is **repeated allocation of large temporary arrays during preview processing**.

---

## Detailed Findings

### 1. Preview Worker Pattern - ✅ CONFIRMED

**Location**: `divere/core/app_context.py`

```python
# Lines 25-55: _PreviewWorker spawns in QThreadPool
class _PreviewWorker(QRunnable):
    def run(self):
        result_image = self.the_enlarger.apply_full_pipeline(image, params, ...)
        result_image = self.color_space_manager.convert_to_display_space(result_image, "DisplayP3")
```

**Finding**: Each slider drag triggers `_trigger_preview_update()` → spawns `_PreviewWorker` → processes entire preview pipeline on proxy image.

**Impact**: Repeated preview calls with slightly varying parameters cause varying memory peaks.

---

### 2. Array Allocation Pattern - ✅ CONFIRMED (Critical Issue)

#### Multiple Copies Per Preview

**Pipeline Processor** (`pipeline_processor.py:400`):
```python
working_array = image.array.copy()  # Copy #1: ~48MB for 2000x2000 proxy
```

**Math Operations** (`math_ops.py:1332`):
```python
result_array = image_array.copy()  # Copy #2: Another ~48MB
```

**NumPy Operations** (throughout codebase):
```python
# color_space.py:756 - Every operation creates new array
return np.power(image_array, gamma)  # Copy #3, #4, #5...

# math_ops.py - No in-place operations
density_array = self.linear_to_density(result_array)  # New array
density_array = self.apply_density_matrix(density_array, ...)  # New array
```

**Finding**:
- **58 occurrences** of `np.power()`, `np.clip()`, `np.dot()` across core modules
- **Zero usage** of NumPy's `out=` parameter for in-place operations
- Each operation creates a new array instead of reusing buffers

**Memory Impact Calculation** (for 2000×2000 proxy):
```
Base image:           48 MB
Pipeline copy:        48 MB
Math pipeline copy:   48 MB
NumPy temporaries:    ~100-200 MB (density conversion, matrix ops, gamma, etc.)
Color space convert:  ~50 MB
Display gamma:        ~48 MB
─────────────────────────
Peak per preview:     ~350-400 MB
```

**This matches the stair-step pattern**: Each preview with slightly different parameters may trigger different code paths, causing the allocator to request more heap pages.

---

### 3. Cache Management - ⚠️ DIFFERENT FROM MEMORY.MD ASSUMPTIONS

**ImageManager** (`image_manager.py:36`):
```python
self._max_cache_size = 1  # Only 1 proxy cached!
```

**LUTProcessor** (`lut_processor.py:21`):
```python
self._max_cache_size = 1  # Only 1 LUT cached!
```

**Finding**: Caches are **ALREADY VERY LIMITED** (size=1), not unbounded as suggested in `memory.md`. Both use correct `OrderedDict` LRU with explicit cleanup on eviction.

**Conclusion**: Cache sizes are NOT the problem. The issue is in the **pipeline temporary allocations**, not cache accumulation.

---

### 4. Display Space Conversion - ✅ CONFIRMED

**Location**: `color_space.py:710, 715`

```python
# Line 710: In-place mutation BUT underlying _apply_color_conversion creates new array
image.array = self._apply_color_conversion(image.array, conversion_matrix, gain_vector)

# Line 715: _apply_gamma creates new array via np.power()
image.array = self._apply_gamma(image.array, gamma)
```

**Finding**: Even though assignment looks in-place, both `_apply_color_conversion()` and `_apply_gamma()` return **new arrays**, adding to temporary allocation burden.

---

## Root Cause Analysis

### Why the Stair-Step Pattern Occurs

1. **Functional Programming Style**: The pipeline uses a functional approach where each transformation returns a new array
2. **No Buffer Reuse**: No workspace/buffer pool exists for preview processing
3. **Allocator Behavior**: When a preview needs peak memory > previous peak, the allocator requests more heap from OS
4. **No Memory Return**: When preview completes, arrays are freed at Python level, but allocator keeps the expanded heap as free blocks (doesn't return to OS)
5. **Monotonic Growth**: Each "new high-water mark" permanently expands the process footprint

### What IS Working Well

1. ✅ **Cache limits**: Already set to 1 (very conservative)
2. ✅ **LRU eviction**: Proper `OrderedDict` usage
3. ✅ **Explicit cleanup**: Caches have `clear_cache()` methods
4. ✅ **Worker lifecycle**: `_PreviewWorker` clears references (`self.image = None`) after copying to locals
5. ✅ **View optimization**: `_trigger_preview_update()` uses `.view()` instead of `.copy()` when passing data to worker

---

## Recommended Solutions (In Order of Preference)

### Option 1: Hybrid Approach (RECOMMENDED)

**Phase 1 - Quick Wins (1-2 days)**
- Audit and remove unnecessary `.copy()` calls
- Use NumPy `out=` parameter in hot paths (density ops, gamma, matrix multiply)
- Profile with memory profiler to identify top 5 allocation sites
- **Expected Impact**: 30-40% reduction in peak memory

**Phase 2 - Buffer Pool for Preview Only (1-2 weeks)**
```python
class PreviewWorkspace:
    """Reusable buffers for preview pipeline"""
    def __init__(self, shape: Tuple[int, int, int]):
        self.buf1 = np.empty(shape, dtype=np.float32)
        self.buf2 = np.empty(shape, dtype=np.float32)
        self.buf3 = np.empty(shape, dtype=np.float32)
        self.shape = shape

    def get_buffer(self, index: int) -> np.ndarray:
        return [self.buf1, self.buf2, self.buf3][index]
```

- Store in `ApplicationContext._preview_workspace`
- Pass to `apply_full_pipeline(..., workspace=workspace)` for previews
- Refactor **only preview path** to use ping-pong buffers
- Keep export path unchanged (correctness > performance)
- **Expected Impact**: Preview peak memory stabilizes at ~150-200MB (down from 400MB)

**Phase 3 - Monitoring (3 days)**
- Add memory usage tracking to `ApplicationContext`
- Log memory stats after each preview
- Alert if footprint exceeds threshold (e.g., 2GB growth)
- **Expected Impact**: Early detection of issues

**Advantages**:
- Incremental, low-risk approach
- Maintains code clarity for export path
- Can stop after Phase 1 if sufficient
- Aligns with development guidelines (incremental progress)

**Disadvantages**:
- Requires moderate refactoring effort
- Need to maintain two code paths (preview vs export)

---

### Option 2: Reduce Proxy Resolution (PRAGMATIC FALLBACK)

**Implementation**:
```python
# image_manager.py:324
def generate_proxy(self, image: ImageData, max_size: Tuple[int, int] = (1200, 1200)):  # Down from (2000, 2000)
```

**Calculation**:
```
Old proxy: 2000×2000 = 4M pixels → ~48MB per array × 8 copies = ~400MB peak
New proxy: 1200×1200 = 1.44M pixels → ~17MB per array × 8 copies = ~140MB peak
```

**Advantages**:
- **One-line change**
- Immediate 65% memory reduction
- No architectural changes

**Disadvantages**:
- Lower preview quality (may be unacceptable for users)
- Doesn't address root cause
- Still has stair-step pattern (just smaller steps)

---

### Option 3: Process Isolation (NUCLEAR OPTION)

**From `memory.md` section 5.5**:
- Move preview pipeline to separate worker process
- Kill and restart worker when footprint exceeds threshold
- Only way to truly reset allocator heap

**Advantages**:
- Completely solves heap high-water mark issue
- Can reset memory to zero periodically

**Disadvantages**:
- High implementation complexity (IPC, serialization)
- Slower preview response (process spawn + data transfer)
- More potential for bugs (race conditions, zombie processes)
- Overkill for current problem size

**Recommendation**: **Only consider if Option 1 fails**

---

### Option 4: Full In-Place Refactor (FROM MEMORY.MD, NOT RECOMMENDED)

**What `memory.md` proposes**:
- Refactor ALL pipeline operations to use `out=` parameter
- Implement workspace buffer pool
- Ping-pong between 3-4 fixed buffers

**Why I Don't Recommend This**:
1. **Invasive**: Requires rewriting ~1500 lines across 5+ modules
2. **Error-Prone**: Easy to accidentally mutate shared buffers
3. **Harder to Debug**: In-place ops make state harder to track
4. **Against Guidelines**: CLAUDE.md emphasizes "simplicity" and "clear intent over clever code"
5. **Unnecessary**: Option 1 achieves 70-80% of the benefit with 20% of the effort

---

## Comparison with `memory.md` Recommendations

| memory.md Proposal | Status in Current Code | My Assessment |
|-------------------|----------------------|---------------|
| Preview worker pattern exists | ✅ Confirmed (app_context.py) | Correct |
| Pipeline creates multiple copies | ✅ Confirmed (pipeline_processor.py, math_ops.py) | Correct - this IS the main issue |
| Unbounded caches | ❌ Cache size = 1 (very limited) | Outdated assumption |
| Need buffer pool | ⚠️ Agree for preview, not export | Partially agree |
| Full in-place refactor | ❌ Too invasive | Disagree - hybrid approach better |
| LRU cache limits | ✅ Already implemented | Already done |
| Process isolation | ⚠️ Overkill for now | Last resort only |

---

## Key Differences from memory.md

1. **Caches are NOT the problem**: They're already size=1 with proper LRU
2. **Full refactor is overkill**: Hybrid approach achieves most benefit with less risk
3. **Export path should stay unchanged**: Correctness matters more than preview performance
4. **Pragmatic fallback exists**: Simply reducing proxy size gives 65% memory reduction with zero risk

---

## Implementation Priority

### Immediate Actions (This Week)
1. ✅ **Validate with profiling**: Run `memory_profiler` on preview pipeline to confirm findings
2. ✅ **Try pragmatic fix first**: Test with `max_size=(1200, 1200)` to see if acceptable
3. ✅ **Document findings**: Share this report with team

### If Pragmatic Fix Insufficient (Next 1-2 Weeks)
4. Implement Option 1 Phase 1 (quick wins with `out=` parameter)
5. Profile again to measure improvement
6. If still needed, implement Phase 2 (buffer pool for preview)

### Long-term Monitoring (Ongoing)
7. Add memory tracking to ApplicationContext
8. Log memory stats periodically
9. Consider process isolation only if problem persists despite above fixes

---

## Testing Strategy

### Before Changes
```python
# Capture baseline
1. Open large image (e.g., 6000×4000)
2. Generate proxy (2000×2000)
3. Drag 20 different sliders
4. Record Activity Monitor "Memory" at each step
5. Note stair-step pattern and final footprint
```

### After Each Phase
```python
# Measure improvement
1. Repeat same test with same image
2. Compare:
   - Peak memory per preview
   - Final footprint after 20 previews
   - Preview response time (ensure no regression)
3. Success criteria:
   - Phase 1: 30% memory reduction, <5% time increase
   - Phase 2: Footprint stabilizes (no stair-step), <10% time increase
```

---

## Conclusion

**The `memory.md` analysis is fundamentally correct** about the heap high-water mark problem and its root cause (repeated array allocation in preview pipeline). However:

1. **Cache limits are already good** (not the issue)
2. **Full in-place refactor is overkill** (hybrid approach is better)
3. **Pragmatic solutions exist** (reduce proxy size as quick fix)

**Recommended Path**:
1. Try reducing proxy resolution first (1-line change, 65% reduction)
2. If insufficient, implement Option 1 hybrid approach
3. Monitor with memory tracking
4. Only consider process isolation if all else fails

**Key Philosophy** (from CLAUDE.md):
> "Pragmatic over dogmatic - Adapt to project reality"
> "Single responsibility per function/class"
> "If you need to explain it, it's too complex"

The hybrid approach aligns with these principles better than a full in-place refactor.

---

## Phase 2 Implementation TODO

### Status: Phase 1 Complete ✅, Phase 2 In Progress 🚧

**Phase 1 Achievements**:
- ✅ Added `out=` parameter to 8 core functions (math_ops.py, color_space.py)
- ✅ Optimized `apply_density_matrix` to eliminate ~48MB temp allocation
- ✅ Optimized `_apply_matrix_parallel` to use `empty` instead of `zeros`
- ✅ All functions tested and verified (max diff < 1e-6)

**Phase 2 Goals**:
- Create buffer pool for preview pipeline only
- Integrate buffer pool into preview worker
- Measure actual memory reduction
- Target: 50-60% reduction in preview peak memory

---

### TODO: Phase 2 - Buffer Pool Implementation

#### Step 1: Analyze Preview Pipeline Call Chain ⏳
**Goal**: Map complete data flow from `_PreviewWorker` to all array allocations

**Tasks**:
- [ ] Trace `_PreviewWorker.run()` → `the_enlarger.apply_full_pipeline()`
- [ ] Trace `apply_full_pipeline()` → `FilmPipelineProcessor` methods
- [ ] Trace `FilmPipelineProcessor` → `FilmMathOps` methods
- [ ] Trace color space conversion chain
- [ ] Document array shapes at each step (for buffer sizing)
- [ ] Identify minimum number of buffers needed (ping-pong pattern)

**Expected Output**: Call graph with array allocations annotated

---

#### Step 2: Create PreviewWorkspace Class ⏳
**Goal**: Implement reusable buffer pool for preview pipeline

**Location**: `divere/core/preview_workspace.py` (new file)

**Design**:
```python
class PreviewWorkspace:
    """Reusable buffers for preview pipeline to reduce allocations"""

    def __init__(self, shape: Tuple[int, int, int], dtype=np.float32):
        # Allocate 3-4 fixed buffers (ping-pong pattern)
        self.buffer_a = np.empty(shape, dtype=dtype)
        self.buffer_b = np.empty(shape, dtype=dtype)
        self.buffer_c = np.empty(shape, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def get_buffer(self, index: int) -> np.ndarray:
        """Get buffer by index (0, 1, 2)"""
        return [self.buffer_a, self.buffer_b, self.buffer_c][index]

    def resize_if_needed(self, new_shape: Tuple[int, int, int]):
        """Reallocate buffers if shape changed"""
        if new_shape != self.shape:
            self.__init__(new_shape, self.dtype)
```

**Tasks**:
- [ ] Create `divere/core/preview_workspace.py`
- [ ] Implement `PreviewWorkspace` class
- [ ] Add shape validation
- [ ] Add resize capability for different proxy sizes
- [ ] Add unit tests

---

#### Step 3: Integrate into ApplicationContext ⏳
**Goal**: Add workspace to preview worker lifecycle

**Location**: `divere/core/app_context.py`

**Changes**:
```python
class ApplicationContext:
    def __init__(self):
        # ... existing code ...
        self._preview_workspace: Optional[PreviewWorkspace] = None

    def _trigger_preview_update(self):
        # ... existing code ...

        # Ensure workspace matches proxy shape
        proxy_shape = proxy_image.array.shape
        if (self._preview_workspace is None or
            self._preview_workspace.shape != proxy_shape):
            self._preview_workspace = PreviewWorkspace(proxy_shape)

        worker = _PreviewWorker(
            # ... existing params ...
            workspace=self._preview_workspace
        )
```

**Tasks**:
- [ ] Add `_preview_workspace` field to `ApplicationContext`
- [ ] Initialize workspace lazily on first preview
- [ ] Resize workspace when proxy shape changes
- [ ] Pass workspace to `_PreviewWorker`
- [ ] Add memory tracking/logging (optional)

---

#### Step 4: Modify Preview Pipeline to Use Buffers ⏳
**Goal**: Update preview code path to use buffer pool

**Location**: Multiple files (pipeline_processor.py, math_ops.py callers)

**Strategy**: Ping-pong pattern
```python
# Example: FilmPipelineProcessor.process_preview()
def process_preview(self, image, params, workspace):
    buf_idx = 0  # Start with buffer A

    # Step 1: Linear to density (write to buffer 0)
    result = self.math_ops.linear_to_density(
        image.array, out=workspace.get_buffer(buf_idx)
    )
    buf_idx = 1 - buf_idx  # Switch to buffer 1

    # Step 2: Apply matrix (read from buffer 0, write to buffer 1)
    result = self.math_ops.apply_density_matrix(
        result, matrix, dmax, out=workspace.get_buffer(buf_idx)
    )
    buf_idx = 1 - buf_idx  # Switch back

    # ... continue ping-pong pattern ...
```

**Tasks**:
- [ ] Add `workspace` parameter to `_PreviewWorker.run()`
- [ ] Add `workspace` parameter to `the_enlarger.apply_full_pipeline()`
- [ ] Add `workspace` parameter to `FilmPipelineProcessor` methods
- [ ] Implement ping-pong buffer usage in pipeline
- [ ] Ensure export path remains unchanged (workspace=None)
- [ ] Add assertions to catch misuse

---

#### Step 5: Testing and Validation ⏳
**Goal**: Verify correctness and measure memory improvement

**Memory Profiling**:
```python
# Test script: test_memory_improvement.py
from memory_profiler import profile
import psutil

@profile
def test_preview_pipeline_old():
    # Load 2000x2000 proxy
    # Trigger 20 previews without buffer pool
    # Record peak memory

@profile
def test_preview_pipeline_new():
    # Load 2000x2000 proxy
    # Trigger 20 previews WITH buffer pool
    # Record peak memory
```

**Tasks**:
- [ ] Create memory profiling test script
- [ ] Run baseline (current code without buffer pool active)
- [ ] Run optimized (with buffer pool)
- [ ] Compare peak memory per preview
- [ ] Verify no stair-step growth pattern
- [ ] Run full test suite to ensure no regressions
- [ ] Visual regression testing (pixel-perfect comparison)
- [ ] Performance benchmarking (ensure <5% slowdown)

**Success Criteria**:
- ✅ Preview memory reduced by 50-60% (400MB → 150-200MB)
- ✅ No visual differences in preview
- ✅ All tests pass
- ✅ Preview latency ≤ 105% of baseline
- ✅ Export quality unchanged

---

#### Step 6: Monitoring and Documentation ⏳
**Goal**: Add observability and document changes

**Tasks**:
- [ ] Add optional memory logging to `ApplicationContext`
- [ ] Log workspace allocation/resize events
- [ ] Document buffer pool usage in code comments
- [ ] Update IMPLEMENTATION_PLAN.md with results
- [ ] Create PR with before/after memory profiles

---

### Risk Mitigation

**Risk 1: Buffer aliasing bugs**
- Mitigation: Careful review of ping-pong logic
- Mitigation: Assertions to detect double-use
- Mitigation: Extensive testing with different images

**Risk 2: Shape mismatches**
- Mitigation: Automatic resize in PreviewWorkspace
- Mitigation: Validation in all `out=` parameters (already implemented)

**Risk 3: Export path affected**
- Mitigation: Only preview path uses workspace
- Mitigation: Export continues to use workspace=None
- Mitigation: Separate test suite for export

---

### Rollback Plan

If Phase 2 fails or causes issues:

1. **Immediate**: Set `workspace=None` in all calls (falls back to Phase 1 behavior)
2. **Partial**: Remove workspace from specific problem functions
3. **Full**: Revert all Phase 2 commits (Phase 1 improvements remain)

Each step is in separate git commit for granular rollback.