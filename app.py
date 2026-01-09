# app.py
# Streamlit demo: "Divide Smart and Conquer" - Tree Visualization
# Run:
#   pip install streamlit
#   streamlit run app.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import random
import streamlit as st
import json
import time
import pandas as pd
import os


# -----------------------------
# Preset Datasets (Real-world scenarios)
# -----------------------------
PRESET_DATASETS = {
    "IoT Sensors - Nhiệt độ từ nhiều cảm biến": {
        "data": [18, 20, 22, 25, 28, 30, 15, 17, 19, 21, 19, 21, 24, 27, 29],
        "description": "Mỗi cảm biến đo nhiệt độ tăng dần trong ngày, khi chuyển sang cảm biến khác thì reset"
    },
    "Banking - Lịch sử giao dịch số dư": {
        "data": [100, 150, 200, 250, 230, 210, 180, 150, 180, 200, 250, 300],
        "description": "Nạp tiền (tăng) và rút tiền (giảm) liên tục"
    },
    "Stock Market - Giá cổ phiếu": {
        "data": [100, 105, 110, 115, 112, 108, 104, 100, 105, 110, 115, 120],
        "description": "Xu hướng tăng -> điều chỉnh giảm -> phục hồi"
    },
    "Student Scores - Điểm thi xen kẽ": {
        "data": [7, 8, 9, 5, 4, 3, 8, 9, 10, 6, 5, 4],
        "description": "Môn dễ (điểm cao) xen kẽ môn khó (điểm thấp)"
    },
    "E-commerce - Giá sản phẩm theo mùa": {
        "data": [50, 60, 70, 80, 75, 70, 65, 60, 70, 80, 90, 100],
        "description": "Tăng giá đầu mùa, giảm giữa mùa, tăng lại cuối mùa"
    },
    "Credit Card - Số tiền giao dịch (từ CSV)": {
        "data": "csv",  # Load from CSV file
        "description": "Dữ liệu giao dịch thẻ tín dụng từ file creditcard.csv (cột Amount)"
    },
    "Random - Mảng ngẫu nhiên": {
        "data": [],  # Will be generated
        "description": "Mảng số ngẫu nhiên để test"
    }
}


# -----------------------------
# Cache Functions for Credit Card Dataset
# -----------------------------
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "creditcard_results.json")

def get_cache_key(arr: List[int]) -> str:
    """Generate a cache key based on array length and hash"""
    import hashlib
    arr_str = ",".join(map(str, arr[:100]))  # Use first 100 elements for hash
    return f"{len(arr)}_{hashlib.md5(arr_str.encode()).hexdigest()[:8]}"

def save_comparison_cache(arr: List[int], results: Dict[str, Any]) -> bool:
    """Save comparison results to cache file"""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        cache_key = get_cache_key(arr)
        
        # Convert results to serializable format (exclude tree objects)
        cache_data = {
            "cache_key": cache_key,
            "array_length": len(arr),
            "array_sample": arr[:50],  # Store first 50 elements for reference
            "results": {}
        }
        
        for algo_name, result in results.items():
            cache_data["results"][algo_name] = {
                "time": result["time"],
                "metrics": result["metrics"],
                "sorted": result["sorted"][:100] if result["sorted"] else [],  # Store first 100 sorted
                "sorted_length": len(result["sorted"]) if result["sorted"] else 0
            }
        
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving cache: {e}")
        return False

def load_comparison_cache(arr: List[int]) -> Optional[Dict[str, Any]]:
    """Load comparison results from cache if exists and matches"""
    try:
        if not os.path.exists(CACHE_FILE):
            return None
        
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check if cache matches current array
        cache_key = get_cache_key(arr)
        if cache_data.get("cache_key") != cache_key:
            return None
        
        if cache_data.get("array_length") != len(arr):
            return None
        
        return cache_data["results"]
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None

def clear_cache():
    """Clear the cache file"""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
    return False


# -----------------------------
# Tree Node Structure
# -----------------------------
@dataclass
class TreeNode:
    """Node in the merge sort tree"""
    array: List[int]
    start_idx: int
    end_idx: int
    level: int
    is_sorted: bool = False
    left: Optional[TreeNode] = None
    right: Optional[TreeNode] = None
    merged_result: Optional[List[int]] = None
    node_id: int = 0


# -----------------------------
# Global counter for node IDs
# -----------------------------
_node_counter = 0

def get_node_id():
    global _node_counter
    _node_counter += 1
    return _node_counter

def reset_node_counter():
    global _node_counter
    _node_counter = 0


# -----------------------------
# Metrics Calculation
# -----------------------------
def calculate_metrics(steps: List[Dict]) -> Dict[str, Any]:
    """Extract performance metrics from algorithm steps"""
    metrics = {
        "total_steps": len(steps),
        "num_runs": 0,
        "num_merges": 0,
        "num_carries": 0,
        "num_flushes": 0,
        "max_run_length": 0,
        "min_run_length": float('inf'),
        "avg_run_length": 0
    }
    
    run_lengths = []
    
    for step in steps:
        step_type = step.get("type", "")
        
        if step_type == "detect_run":
            metrics["num_runs"] += 1
            run = step.get("run", [])
            if run:
                run_lengths.append(len(run))
        elif step_type == "merge" or step_type == "merge_inv":
            metrics["num_merges"] += 1
        elif step_type == "carry":
            metrics["num_carries"] += 1
        elif step_type == "flush_block":
            metrics["num_flushes"] += 1
    
    if run_lengths:
        metrics["max_run_length"] = max(run_lengths)
        metrics["min_run_length"] = min(run_lengths)
        metrics["avg_run_length"] = sum(run_lengths) / len(run_lengths)
    else:
        metrics["min_run_length"] = 0
    
    return metrics


# -----------------------------
# Pure Benchmark Functions (No Logging - for accurate timing)
# -----------------------------
def benchmark_alg1(arr: List[int]) -> Tuple[List[int], int, int]:
    """Algorithm 1 - pure benchmark without logging. Returns (sorted_arr, num_runs, num_merges)"""
    if len(arr) <= 1:
        return arr[:], 1, 0
    
    # Find runs
    runs = []
    cur_run = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] >= arr[i-1]:
            cur_run.append(arr[i])
        else:
            runs.append(cur_run[:])
            cur_run = [arr[i]]
    runs.append(cur_run)
    
    num_runs = len(runs)
    num_merges = 0
    
    # Tournament merge
    while len(runs) > 1:
        next_level = []
        for i in range(0, len(runs), 2):
            if i + 1 < len(runs):
                merged = merge_sorted(runs[i], runs[i+1])
                next_level.append(merged)
                num_merges += 1
            else:
                next_level.append(runs[i])
        runs = next_level
    
    return runs[0], num_runs, num_merges


def benchmark_alg2(arr: List[int]) -> Tuple[List[int], int, int]:
    """Algorithm 2 - pure benchmark without logging. Returns (sorted_arr, num_runs, num_merges)"""
    if len(arr) <= 1:
        return arr[:], 1, 0
    
    # Find monotonic runs
    runs = []
    i = 0
    
    while i < len(arr):
        if i == len(arr) - 1:
            runs.append([arr[i]])
            break
        
        # Detect direction
        j = i
        while j + 1 < len(arr) and arr[j+1] == arr[j]:
            j += 1
        
        if j + 1 < len(arr):
            is_inc = arr[j+1] > arr[j]
        else:
            is_inc = True
        
        # Build run
        run = [arr[i]]
        k = i
        while k + 1 < len(arr):
            if is_inc and arr[k+1] >= arr[k]:
                run.append(arr[k+1])
                k += 1
            elif not is_inc and arr[k+1] <= arr[k]:
                run.append(arr[k+1])
                k += 1
            else:
                break
        
        # Normalize if decreasing
        if not is_inc and len(run) >= 2 and run[0] > run[-1]:
            run = list(reversed(run))
        
        runs.append(run)
        i = k + 1
    
    num_runs = len(runs)
    num_merges = 0
    
    # Tournament merge
    while len(runs) > 1:
        next_level = []
        for i in range(0, len(runs), 2):
            if i + 1 < len(runs):
                merged = merge_sorted(runs[i], runs[i+1])
                next_level.append(merged)
                num_merges += 1
            else:
                next_level.append(runs[i])
        runs = next_level
    
    return runs[0], num_runs, num_merges


def benchmark_alg3(arr: List[int]) -> Tuple[List[int], int, int]:
    """Algorithm 3 - pure benchmark without logging. Returns (sorted_arr, num_runs, num_merges)"""
    if len(arr) <= 1:
        return arr[:], 1, 0
    
    runs_inc = []
    L = []
    R = []
    inc = float("-inf")
    dec = float("inf")
    num_merge_inv = 0
    
    def flush_block():
        nonlocal L, R, inc, dec, runs_inc, num_merge_inv
        if not L and not R:
            return
        
        # Merge L (increasing) and R (decreasing)
        result = []
        i = 0
        j = len(R) - 1
        
        while i < len(L) and j >= 0:
            if L[i] <= R[j]:
                result.append(L[i])
                i += 1
            else:
                result.append(R[j])
                j -= 1
        
        result.extend(L[i:])
        while j >= 0:
            result.append(R[j])
            j -= 1
        
        runs_inc.append(result)
        num_merge_inv += 1
        
        L = []
        R = []
        inc = float("-inf")
        dec = float("inf")
    
    i = 0
    while i < len(arr):
        x = arr[i]
        can_L = (x >= inc)
        can_R = (x <= dec)
        next_val = arr[i+1] if i+1 < len(arr) else None
        
        if can_L and can_R:
            if next_val is None or x <= next_val:
                L.append(x)
                inc = x
            else:
                R.append(x)
                dec = x
            i += 1
        elif can_L:
            L.append(x)
            inc = x
            i += 1
        elif can_R:
            R.append(x)
            dec = x
            i += 1
        else:
            flush_block()
    
    flush_block()
    
    num_runs = len(runs_inc)
    num_merges = num_merge_inv
    
    # Tournament merge
    while len(runs_inc) > 1:
        next_level = []
        for i in range(0, len(runs_inc), 2):
            if i + 1 < len(runs_inc):
                merged = merge_sorted(runs_inc[i], runs_inc[i+1])
                next_level.append(merged)
                num_merges += 1
            else:
                next_level.append(runs_inc[i])
        runs_inc = next_level
    
    return runs_inc[0] if runs_inc else [], num_runs, num_merges


# -----------------------------
# Algorithm 1: Consecutive Increasing Runs
# -----------------------------
def build_runs_tree_alg1(arr: List[int]) -> Tuple[TreeNode, List[Dict]]:
    """Build tree for Algorithm 1 - consecutive increasing runs"""
    reset_node_counter()
    steps = []
    
    if len(arr) <= 1:
        root = TreeNode(arr[:], 0, len(arr)-1 if arr else 0, 0, True, node_id=get_node_id())
        steps.append({
            "type": "trivial",
            "message": "Mảng có độ dài <= 1, đã được sắp xếp",
            "node_id": root.node_id,
            "array": arr[:]
        })
        return root, steps
    
    # Find runs
    runs = []
    cur_run = [arr[0]]
    cur_start = 0
    
    for i in range(1, len(arr)):
        if arr[i] >= arr[i-1]:
            cur_run.append(arr[i])
        else:
            runs.append((cur_run[:], cur_start, i-1))
            steps.append({
                "type": "detect_run",
                "message": f"Phát hiện dãy tăng từ [{cur_start}] đến [{i-1}]: {cur_run}",
                "run": cur_run[:],
                "indices": (cur_start, i-1)
            })
            cur_run = [arr[i]]
            cur_start = i
    
    runs.append((cur_run[:], cur_start, len(arr)-1))
    steps.append({
        "type": "detect_run",
        "message": f"Phát hiện dãy tăng từ [{cur_start}] đến [{len(arr)-1}]: {cur_run}",
        "run": cur_run[:],
        "indices": (cur_start, len(arr)-1)
    })
    
    steps.append({
        "type": "runs_complete",
        "message": f"Đã tách thành {len(runs)} dãy tăng",
        "runs": [r[0] for r in runs]
    })
    
    # Build tree from runs using tournament merge
    nodes = []
    for r in runs:
        node = TreeNode(r[0], r[1], r[2], 0, True, node_id=get_node_id())
        nodes.append(node)
        steps.append({
            "type": "create_leaf",
            "message": f"Tạo dãy con: {r[0]}",
            "array": r[0],
            "node_id": node.node_id
        })
    
    # Track all nodes for visualization
    all_nodes = nodes[:]
    
    level = 1
    while len(nodes) > 1:
        next_level = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                # Merge two nodes
                left = nodes[i]
                right = nodes[i+1]
                merged = merge_sorted(left.array, right.array)
                
                parent = TreeNode(
                    merged,
                    left.start_idx,
                    right.end_idx,
                    level,
                    True,
                    left,
                    right,
                    merged,
                    get_node_id()
                )
                
                steps.append({
                    "type": "merge",
                    "message": f"Ghép 2 dãy: {left.array} + {right.array} = {merged}",
                    "left": left.array,
                    "right": right.array,
                    "result": merged,
                    "node_id": parent.node_id
                })
                
                next_level.append(parent)
                all_nodes.append(parent)
            else:
                # Odd node, carry forward - create a copy at new level
                carried_node = TreeNode(
                    nodes[i].array,
                    nodes[i].start_idx,
                    nodes[i].end_idx,
                    level,
                    True,
                    node_id=nodes[i].node_id  # Keep same ID for highlighting
                )
                steps.append({
                    "type": "carry",
                    "message": f"Chuyển dãy lẻ sang vòng tiếp: {nodes[i].array}",
                    "array": nodes[i].array,
                    "node_id": nodes[i].node_id
                })
                next_level.append(carried_node)
                all_nodes.append(carried_node)
        
        nodes = next_level
        level += 1
    
    # Store all_nodes in root for visualization
    root = nodes[0]
    root.merged_result = all_nodes  # Store all nodes here for rendering
    return root, steps


def merge_sorted(left: List[int], right: List[int]) -> List[int]:
    """Simple merge of two sorted arrays"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# -----------------------------
# Algorithm 2: Consecutive Monotonic Runs
# -----------------------------
def build_runs_tree_alg2(arr: List[int]) -> Tuple[TreeNode, List[Dict]]:
    """Build tree for Algorithm 2 - consecutive monotonic runs"""
    reset_node_counter()
    steps = []
    
    if len(arr) <= 1:
        root = TreeNode(arr[:], 0, len(arr)-1 if arr else 0, 0, True, node_id=get_node_id())
        steps.append({
            "type": "trivial",
            "message": "Mảng có độ dài <= 1, đã được sắp xếp",
            "node_id": root.node_id,
            "array": arr[:]
        })
        return root, steps
    
    # Find monotonic runs
    runs = []  # (original_run, normalized_run, start, end, direction)
    i = 0
    
    while i < len(arr):
        if i == len(arr) - 1:
            runs.append(([arr[i]], [arr[i]], i, i, "inc"))
            steps.append({
                "type": "detect_run",
                "message": f"Phần tử cuối [{i}]: {arr[i]}",
                "run": [arr[i]],
                "direction": "inc"
            })
            break
        
        # Detect direction
        j = i
        while j + 1 < len(arr) and arr[j+1] == arr[j]:
            j += 1
        
        if j + 1 < len(arr):
            dirn = "inc" if arr[j+1] > arr[j] else "dec"
        else:
            dirn = "inc"
        
        # Build run
        run = [arr[i]]
        k = i
        while k + 1 < len(arr):
            if dirn == "inc" and arr[k+1] >= arr[k]:
                run.append(arr[k+1])
                k += 1
            elif dirn == "dec" and arr[k+1] <= arr[k]:
                run.append(arr[k+1])
                k += 1
            else:
                break
        
        # Normalize if decreasing
        if dirn == "dec" and len(run) >= 2 and run[0] > run[-1]:
            run_normalized = list(reversed(run))
            steps.append({
                "type": "detect_run",
                "message": f"Phát hiện dãy giảm [{i}→{k}]: {run}",
                "run": run[:],
                "normalized": run_normalized,
                "direction": "dec"
            })
            runs.append((run[:], run_normalized, i, k, "dec"))
        else:
            steps.append({
                "type": "detect_run",
                "message": f"Phát hiện dãy tăng [{i}→{k}]: {run}",
                "run": run[:],
                "direction": "inc"
            })
            runs.append((run[:], run[:], i, k, "inc"))
        
        i = k + 1
    
    steps.append({
        "type": "runs_complete",
        "message": f"Đã tách thành {len(runs)} dãy đơn điệu",
        "runs": [r[0] for r in runs]
    })
    
    # Build tree from runs - display original, use normalized for merging
    nodes = []
    for r in runs:
        original_run = r[0]
        normalized_run = r[1]
        # Create node with ORIGINAL run for display, but store normalized for merging
        node = TreeNode(original_run, r[2], r[3], 0, True, node_id=get_node_id())
        node.merged_result = normalized_run  # Store normalized for merging
        nodes.append(node)
        steps.append({
            "type": "create_leaf",
            "message": f"Tạo dãy con: {original_run} ({'giảm' if r[4] == 'dec' else 'tăng'})",
            "array": original_run,
            "node_id": node.node_id,
            "direction": r[4]
        })
    
    # Track all nodes for visualization
    all_nodes = nodes[:]
    
    level = 1
    while len(nodes) > 1:
        next_level = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                left = nodes[i]
                right = nodes[i+1]
                # Use normalized arrays for merging
                left_merge = left.merged_result if left.merged_result else left.array
                right_merge = right.merged_result if right.merged_result else right.array
                merged = merge_sorted(left_merge, right_merge)
                
                parent = TreeNode(
                    merged,
                    left.start_idx,
                    right.end_idx,
                    level,
                    True,
                    left,
                    right,
                    merged,
                    get_node_id()
                )
                
                steps.append({
                    "type": "merge",
                    "message": f"Ghép: {left.array} + {right.array} = {merged}",
                    "left": left.array,
                    "right": right.array,
                    "result": merged,
                    "node_id": parent.node_id
                })
                
                next_level.append(parent)
                all_nodes.append(parent)
            else:
                # Odd node, carry forward - create a copy at new level
                carried_node = TreeNode(
                    nodes[i].array,
                    nodes[i].start_idx,
                    nodes[i].end_idx,
                    level,
                    True,
                    node_id=nodes[i].node_id  # Keep same ID for highlighting
                )
                # Preserve the normalized data for merging
                carried_node.merged_result = nodes[i].merged_result
                steps.append({
                    "type": "carry",
                    "message": f"Chuyển dãy lẻ sang vòng tiếp: {nodes[i].array}",
                    "array": nodes[i].array,
                    "node_id": nodes[i].node_id
                })
                next_level.append(carried_node)
                all_nodes.append(carried_node)
        
        nodes = next_level
        level += 1
    
    # Store all_nodes in root for visualization
    root = nodes[0]
    root.merged_result = all_nodes
    return root, steps


# -----------------------------
# Algorithm 3: Non-consecutive L/R
# -----------------------------
def build_runs_tree_alg3(arr: List[int]) -> Tuple[TreeNode, List[Dict]]:
    """Build tree for Algorithm 3 - non-consecutive alternating L/R"""
    reset_node_counter()
    steps = []
    
    if len(arr) <= 1:
        root = TreeNode(arr[:], 0, len(arr)-1 if arr else 0, 0, True, node_id=get_node_id())
        steps.append({
            "type": "trivial",
            "message": "Mảng có độ dài <= 1, đã được sắp xếp",
            "node_id": root.node_id,
            "array": arr[:]
        })
        return root, steps
    
    runs_inc = []
    L = []
    R = []
    inc = float("-inf")
    dec = float("inf")
    
    def flush_block():
        nonlocal L, R, inc, dec, runs_inc
        if not L and not R:
            return
        
        steps.append({
            "type": "flush_block",
            "message": f"Đẩy khối: L={L}, R={R}",
            "L": L[:],
            "R": R[:]
        })
        
        merged = merge_inv(L, R)
        runs_inc.append(merged)
        
        steps.append({
            "type": "merge_inv",
            "message": f"Merge_Inv: L(tăng)={L} + R(giảm)={R} = {merged}",
            "result": merged
        })
        
        L = []
        R = []
        inc = float("-inf")
        dec = float("inf")
    
    i = 0
    while i < len(arr):
        x = arr[i]
        can_L = (x >= inc)
        can_R = (x <= dec)
        next_val = arr[i+1] if i+1 < len(arr) else None
        
        if can_L and can_R:
            # Heuristic: nếu x <= next_val thì vào L, ngược lại vào R
            # Nếu là phần tử cuối (không có next_val), ưu tiên vào L
            if next_val is None:
                # Phần tử cuối cùng, ưu tiên vào L (dãy tăng)
                L.append(x)
                inc = x
                steps.append({
                    "type": "put_L",
                    "message": f"A[{i}]={x} → L (phần tử cuối, ưu tiên L)",
                    "element": x,
                    "L": L[:],
                    "R": R[:]
                })
                i += 1
            elif x <= next_val:
                L.append(x)
                inc = x
                steps.append({
                    "type": "put_L",
                    "message": f"A[{i}]={x} → L (heuristic: {x} <= {next_val})",
                    "element": x,
                    "L": L[:],
                    "R": R[:]
                })
                i += 1
            else:
                R.append(x)
                dec = x
                steps.append({
                    "type": "put_R",
                    "message": f"A[{i}]={x} → R (heuristic: {x} > {next_val})",
                    "element": x,
                    "L": L[:],
                    "R": R[:]
                })
                i += 1
        elif can_L and not can_R:
            L.append(x)
            inc = x
            steps.append({
                "type": "put_L",
                "message": f"A[{i}]={x} → L (chỉ vào được L)",
                "element": x,
                "L": L[:],
                "R": R[:]
            })
            i += 1
        elif can_R and not can_L:
            R.append(x)
            dec = x
            steps.append({
                "type": "put_R",
                "message": f"A[{i}]={x} → R (chỉ vào được R)",
                "element": x,
                "L": L[:],
                "R": R[:]
            })
            i += 1
        else:
            flush_block()
    
    flush_block()
    
    steps.append({
        "type": "runs_complete",
        "message": f"Đã xây dựng {len(runs_inc)} dãy tăng",
        "runs": runs_inc
    })
    
    # Build tree from runs
    nodes = []
    for r in runs_inc:
        node = TreeNode(r, 0, 0, 0, True, node_id=get_node_id())
        nodes.append(node)
        steps.append({
            "type": "create_leaf",
            "message": f"Tạo dãy con: {r}",
            "array": r,
            "node_id": node.node_id
        })
    
    # Track all nodes for visualization
    all_nodes = nodes[:]
    
    level = 1
    while len(nodes) > 1:
        next_level = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                left = nodes[i]
                right = nodes[i+1]
                merged = merge_sorted(left.array, right.array)
                
                parent = TreeNode(
                    merged,
                    0,
                    0,
                    level,
                    True,
                    left,
                    right,
                    merged,
                    get_node_id()
                )
                
                steps.append({
                    "type": "merge",
                    "message": f"Ghép: {left.array} + {right.array} = {merged}",
                    "left": left.array,
                    "right": right.array,
                    "result": merged,
                    "node_id": parent.node_id
                })
                
                next_level.append(parent)
                all_nodes.append(parent)
            else:
                # Odd node, carry forward - create a copy at new level
                carried_node = TreeNode(
                    nodes[i].array,
                    nodes[i].start_idx,
                    nodes[i].end_idx,
                    level,
                    True,
                    node_id=nodes[i].node_id  # Keep same ID for highlighting
                )
                # Preserve the normalized data for merging
                carried_node.merged_result = nodes[i].merged_result
                steps.append({
                    "type": "carry",
                    "message": f"Chuyển dãy lẻ sang vòng tiếp: {nodes[i].array}",
                    "array": nodes[i].array,
                    "node_id": nodes[i].node_id
                })
                next_level.append(carried_node)
                all_nodes.append(carried_node)
        
        nodes = next_level
        level += 1
    
    # Store all_nodes in root for visualization
    root = nodes[0]
    root.merged_result = all_nodes
    return root, steps


def merge_inv(L_inc: List[int], R_dec: List[int]) -> List[int]:
    """Merge L(increasing) and R(decreasing) into increasing"""
    result = []
    i = 0
    j = len(R_dec) - 1
    
    while i < len(L_inc) and j >= 0:
        if L_inc[i] <= R_dec[j]:
            result.append(L_inc[i])
            i += 1
        else:
            result.append(R_dec[j])
            j -= 1
    
    result.extend(L_inc[i:])
    while j >= 0:
        result.append(R_dec[j])
        j -= 1
    
    return result


# -----------------------------
# Tree Visualization
# -----------------------------
def render_original_array_with_highlight(arr: List[int], highlight_indices: List[int], step_message: str = "") -> str:
    """Render original array with highlighted elements being processed"""
    html = '<div style="margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 8px;">'
    html += '<div style="font-weight: bold; margin-bottom: 10px; color: #333;">📊 Mảng gốc:</div>'
    
    # Display array with indices
    html += '<div style="display: flex; flex-wrap: wrap; gap: 2px;">'
    for i, val in enumerate(arr):
        is_highlighted = i in highlight_indices
        bg_color = "#ffeb3b" if is_highlighted else "#fff"
        border_color = "#ff5722" if is_highlighted else "#333"
        border_width = "3px" if is_highlighted else "1px"
        box_shadow = "box-shadow: 0 0 10px rgba(255, 87, 34, 0.5);" if is_highlighted else ""
        idx_bg = "#ff9800" if is_highlighted else "#ddd"
        idx_color = "#fff" if is_highlighted else "#666"
        font_weight = "bold" if is_highlighted else "normal"
        
        html += f'<div style="border: {border_width} solid {border_color}; background-color: {bg_color}; font-family: monospace; min-width: 40px; text-align: center; border-radius: 4px; overflow: hidden; {box_shadow}">'
        html += f'<div style="background-color: {idx_bg}; padding: 2px 4px; font-size: 10px; color: {idx_color}; border-bottom: 1px solid {border_color}; font-weight: {font_weight};">{i}</div>'
        html += f'<div style="padding: 8px 12px; font-size: 14px; font-weight: {font_weight};">{val}</div>'
        html += '</div>'
    
    html += '</div>'
    
    if step_message:
        html += f'<div style="margin-top: 10px; color: #666; font-size: 13px;">➤ {step_message}</div>'
    
    html += '</div>'
    return html


def render_tree_html_alg3(current_step: int, steps: List[Dict], arr: List[int], width: int = 1200) -> str:
    """Render Algorithm 3 visualization - L/R blocks building process"""
    
    html = f'<div style="width: {width}px; margin: 20px auto; font-family: monospace;">'
    
    # Extract highlight indices from current step
    highlight_indices = []
    if current_step < len(steps):
        step = steps[current_step]
        # Try to extract index from message like "A[8]=94"
        import re
        match = re.search(r'A\[(\d+)\]', step.get('message', ''))
        if match:
            highlight_indices = [int(match.group(1))]
    
    # Display original array with highlighting
    if arr and current_step < len(steps):
        html += render_original_array_with_highlight(arr, highlight_indices, steps[current_step].get('message', ''))
    
    if current_step >= len(steps):
        current_step = len(steps) - 1
    
    step = steps[current_step]
    step_type = step.get("type", "")
    
    # Section 1: L/R Building Process
    if step_type in ["put_L", "put_R", "flush_block", "merge_inv"]:
        html += f'''<div style="margin: 20px 0;">
            <div style="
                background-color: #FF9800;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 10px;
                display: inline-block;
            ">🔧 Bước 1: Xây dựng L/R Blocks</div>
        '''
        
        # Display current L and R
        html += '<div style="display: flex; gap: 20px; margin: 20px 0;">'
        
        # L (increasing)
        L = step.get("L", [])
        html += f'''<div style="flex: 1; border: 3px solid #4CAF50; border-radius: 8px; padding: 15px; background: #f0fff0;">
            <div style="font-weight: bold; margin-bottom: 10px; color: #4CAF50;">📈 L (Dãy tăng)</div>
            <div style="display: flex; gap: 2px; flex-wrap: wrap;">'''
        
        if L:
            for val in L[:20]:
                html += f'''<div style="
                    border: 1px solid #333;
                    padding: 6px 10px;
                    background: white;
                    font-size: 14px;
                    min-width: 30px;
                    text-align: center;
                    border-radius: 3px;
                ">{val}</div>'''
            if len(L) > 20:
                html += '<div style="padding: 6px;">...</div>'
        else:
            html += '<div style="color: #999; padding: 6px;">Rỗng</div>'
        
        html += '</div></div>'
        
        # R (decreasing)
        R = step.get("R", [])
        html += f'''<div style="flex: 1; border: 3px solid #F44336; border-radius: 8px; padding: 15px; background: #fff0f0;">
            <div style="font-weight: bold; margin-bottom: 10px; color: #F44336;">📉 R (Dãy giảm)</div>
            <div style="display: flex; gap: 2px; flex-wrap: wrap;">'''
        
        if R:
            for val in R[:20]:
                html += f'''<div style="
                    border: 1px solid #333;
                    padding: 6px 10px;
                    background: white;
                    font-size: 14px;
                    min-width: 30px;
                    text-align: center;
                    border-radius: 3px;
                ">{val}</div>'''
            if len(R) > 20:
                html += '<div style="padding: 6px;">...</div>'
        else:
            html += '<div style="color: #999; padding: 6px;">Rỗng</div>'
        
        html += '</div></div>'
        html += '</div>'
        
        # Show merge result if this is merge_inv step
        if step_type == "merge_inv":
            result = step.get("result", [])
            html += f'''<div style="margin: 20px 0; border: 3px solid #2196F3; border-radius: 8px; padding: 15px; background: #e3f2fd;">
                <div style="font-weight: bold; margin-bottom: 10px; color: #2196F3;">🔀 Kết quả Merge_Inv</div>
                <div style="display: flex; gap: 2px; flex-wrap: wrap;">'''
            
            for val in result[:20]:
                html += f'''<div style="
                    border: 1px solid #333;
                    padding: 6px 10px;
                    background: white;
                    font-size: 14px;
                    min-width: 30px;
                    text-align: center;
                    border-radius: 3px;
                ">{val}</div>'''
            
            if len(result) > 20:
                html += '<div style="padding: 6px;">...</div>'
            
            html += '</div></div>'
        
        html += '</div>'
    
    # Section 2: Show all runs created so far
    runs_so_far = []
    for i in range(current_step + 1):
        if i < len(steps) and steps[i].get("type") == "merge_inv":
            runs_so_far.append(steps[i].get("result", []))
    
    if runs_so_far:
        html += f'''<div style="margin: 20px 0;">
            <div style="
                background-color: #2196F3;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 10px;
                display: inline-block;
            ">📦 Các dãy đã tạo ({len(runs_so_far)} runs)</div>
        '''
        
        html += '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0;">'
        
        for idx, run in enumerate(runs_so_far):
            html += f'''<div style="
                border: 2px solid #666;
                border-radius: 8px;
                padding: 10px;
                background-color: #f0f0f0;
                min-width: 100px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 10px; color: #666; margin-bottom: 5px;">Run {idx+1}</div>
                <div style="display: flex; gap: 2px; justify-content: center;">'''
            
            for val in run[:10]:
                html += f'''<div style="
                    border: 1px solid #333;
                    padding: 4px 8px;
                    background: white;
                    font-size: 12px;
                    min-width: 25px;
                    text-align: center;
                ">{val}</div>'''
            
            if len(run) > 10:
                html += '<div style="padding: 4px;">...</div>'
            
            html += '</div></div>'
        
        html += '</div></div>'
    
    # Section 3: Tournament merge visualization if we've reached that stage
    merge_steps = [s for s in steps if s.get("type") in ["merge", "carry", "create_leaf"]]
    if current_step >= 0 and any(s.get("type") in ["merge", "carry"] for i, s in enumerate(steps) if i <= current_step):
        # Find all merge-related steps up to current
        current_merge_idx = 0
        for i in range(current_step + 1):
            if i < len(steps) and steps[i].get("type") in ["merge", "carry"]:
                current_merge_idx = i
        
        # Display tournament merge tree for current state
        html += f'''<div style="margin: 20px 0;">
            <div style="
                background-color: #9C27B0;
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 10px;
                display: inline-block;
            ">🌳 Bước 2: Tournament Merge</div>
        </div>'''
        
        # Show the merge step details
        if current_step < len(steps):
            current_step_data = steps[current_step]
            if current_step_data.get("type") == "merge":
                left = current_step_data.get("left", [])
                right = current_step_data.get("right", [])
                result = current_step_data.get("result", [])
                
                html += '<div style="display: flex; gap: 20px; align-items: center; margin: 20px 0;">'
                
                # Left array
                html += '<div style="flex: 1; border: 2px solid #666; border-radius: 8px; padding: 10px; background: #fff9e6;">'
                html += '<div style="display: flex; gap: 2px; justify-content: center;">'
                for val in left[:10]:
                    html += f'<div style="border: 1px solid #333; padding: 4px 8px; background: white; font-size: 12px; min-width: 25px; text-align: center;">{val}</div>'
                if len(left) > 10:
                    html += '<div>...</div>'
                html += '</div></div>'
                
                # Merge symbol
                html += '<div style="font-size: 24px; color: #9C27B0;">+</div>'
                
                # Right array
                html += '<div style="flex: 1; border: 2px solid #666; border-radius: 8px; padding: 10px; background: #fff9e6;">'
                html += '<div style="display: flex; gap: 2px; justify-content: center;">'
                for val in right[:10]:
                    html += f'<div style="border: 1px solid #333; padding: 4px 8px; background: white; font-size: 12px; min-width: 25px; text-align: center;">{val}</div>'
                if len(right) > 10:
                    html += '<div>...</div>'
                html += '</div></div>'
                
                # Arrow
                html += '<div style="font-size: 24px; color: #9C27B0;">→</div>'
                
                # Result
                html += '<div style="flex: 2; border: 3px solid #9C27B0; border-radius: 8px; padding: 10px; background: #f3e5f5;">'
                html += '<div style="display: flex; gap: 2px; justify-content: center;">'
                for val in result[:10]:
                    html += f'<div style="border: 1px solid #333; padding: 4px 8px; background: white; font-size: 12px; min-width: 25px; text-align: center;">{val}</div>'
                if len(result) > 10:
                    html += '<div>...</div>'
                html += '</div></div>'
                
                html += '</div>'
    
    html += '</div>'
    
    return html


def render_tree_html(node: TreeNode, current_step: int, steps: List[Dict], arr: List[int], width: int = 1200) -> str:
    """Render tree as HTML/CSS visualization - bottom to top (runs at bottom, result at top)"""
    
    html = f'<div style="width: {width}px; margin: 20px auto; font-family: monospace;">'
    
    # Extract highlight indices from current step
    highlight_indices = []
    if current_step < len(steps):
        step = steps[current_step]
        # Check for 'indices' tuple (start, end) in detect_run steps
        if 'indices' in step:
            start, end = step['indices']
            highlight_indices = list(range(start, end + 1))
        else:
            # Try to extract from message
            import re
            match = re.search(r'\[(\d+)\]', step.get('message', ''))
            if match:
                highlight_indices = [int(match.group(1))]
    
    # Display original array with highlighting
    if arr and current_step < len(steps):
        html += render_original_array_with_highlight(arr, highlight_indices, steps[current_step].get('message', ''))
    
    # Get all nodes from root.merged_result (stored during build)
    all_nodes = node.merged_result if isinstance(node.merged_result, list) else []
    if not all_nodes:
        # Fallback: just show root
        all_nodes = [node]
    
    # Group nodes by level
    nodes_by_level = {}
    for n in all_nodes:
        level = n.level
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(n)
    
    if not nodes_by_level:
        nodes_by_level[0] = [node]
    
    depth = max(nodes_by_level.keys()) + 1 if nodes_by_level else 1
    
    # Render each level from 0 to depth-1 (leaves first, then merges, then root)
    for level in range(depth):
        if level not in nodes_by_level:
            continue
        
        nodes_in_level = nodes_by_level[level]
        num_nodes = len(nodes_in_level)
        
        # Determine label for this level
        if level == 0:
            level_label = "🔻 Bước 1: Chia thành các dãy con"
            label_color = "#2196F3"
        elif level == depth - 1:
            level_label = "✅ Kết quả cuối: Mảng đã sắp xếp"
            label_color = "#4CAF50"
        else:
            merge_round = level
            level_label = f"🔄 Bước {merge_round + 1}: Ghép các dãy (vòng {merge_round})"
            label_color = "#FF9800"
        
        # Level header with label
        html += f'''<div style="margin: 20px 0;">
            <div style="
                background-color: {label_color};
                color: white;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 10px;
                display: inline-block;
            ">{level_label}</div>
        '''
        
        html += f'<div style="display: flex; justify-content: space-around; margin: 15px 0; position: relative; flex-wrap: wrap; gap: 10px;">'
        
        for node_obj in nodes_in_level:
            # Highlight if this node is in current step
            is_current = False
            if current_step < len(steps):
                step = steps[current_step]
                if step.get("node_id") == node_obj.node_id or \
                   (step.get("type") in ["merge", "merge_inv"] and step.get("result") == node_obj.array):
                    is_current = True
            
            bg_color = "#ffffcc" if is_current else "#f0f0f0"
            border_color = "#ff6b6b" if is_current else "#666"
            border_width = "3px" if is_current else "2px"
            
            # Render node
            html += f'''<div style="
                border: {border_width} solid {border_color};
                border-radius: 8px;
                padding: 10px;
                background-color: {bg_color};
                min-width: 100px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">'''
            
            # Render array elements
            html += '<div style="display: flex; gap: 2px; justify-content: center;">'
            for val in node_obj.array[:10]:  # Limit display
                html += f'''<div style="
                    border: 1px solid #333;
                    padding: 4px 8px;
                    background: white;
                    font-size: 12px;
                    min-width: 25px;
                    text-align: center;
                ">{val}</div>'''
            
            if len(node_obj.array) > 10:
                html += '<div style="padding: 4px;">...</div>'
            
            html += '</div>'
            html += '</div>'
        
        html += '</div></div>'
    
    html += '</div>'
    
    return html


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Divide Smart & Conquer - Tree Visualization", layout="wide")

# Initialize session state with default values
if "arr" not in st.session_state:
    # Load default example array on first visit
    st.session_state["arr"] = [18, 20, 22, 25, 28, 30, 15, 17, 19, 21, 19, 21, 24, 27, 29]
if "tree" not in st.session_state:
    st.session_state["tree"] = None
if "steps" not in st.session_state:
    st.session_state["steps"] = None
if "sorted" not in st.session_state:
    st.session_state["sorted"] = None
if "step_idx" not in st.session_state:
    st.session_state["step_idx"] = 0
if "algo_type" not in st.session_state:
    st.session_state["algo_type"] = "alg1"
if "comparison_results" not in st.session_state:
    st.session_state["comparison_results"] = None

st.title('🌳 "Divide Smart and Conquer" - Tree Visualization')
st.caption("Trực quan hóa cây cho 3 thuật toán sắp xếp trong bài báo")

# Sidebar controls
with st.sidebar:
    st.header("🧭 Navigation")
    
    # Page selection
    page = st.radio(
        "Chọn trang",
        ["🔍 Demo chi tiết thuật toán", "📊 So sánh 3 thuật toán"],
        index=0
    )
    
    st.divider()
    st.header("⚙️ Cấu hình")
    
    # Dataset selection
    dataset_choice = st.selectbox(
        "📂 Chọn dataset",
        list(PRESET_DATASETS.keys())
    )
    
    if dataset_choice == "Random - Mảng ngẫu nhiên":
        n = st.number_input("Số phần tử", min_value=2, max_value=50, value=15, step=1)
        vmin = st.number_input("Giá trị min", value=0, step=1)
        vmax = st.number_input("Giá trị max", value=100, step=1)
        seed = st.number_input("Seed", value=42, step=1)
        
        if st.button("🎲 Tạo mảng random", type="primary"):
            random.seed(int(seed))
            arr = [random.randint(int(vmin), int(vmax)) for _ in range(int(n))]
            st.session_state["arr"] = arr
            st.session_state["tree"] = None
            st.session_state["steps"] = None
            st.session_state["step_idx"] = 0
            st.session_state["comparison_results"] = None
    
    elif dataset_choice == "Credit Card - Số tiền giao dịch (từ CSV)":
        # Load from CSV file
        csv_file = "creditcard.csv"
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if "Amount" in df.columns:
                    # Options for loading
                    st.write(f"📊 File có {len(df)} giao dịch")
                    
                    num_rows = st.slider(
                        "Số lượng giao dịch để phân tích",
                        min_value=10,
                        max_value=len(df),
                        value=min(50, len(df)),
                        step=10,
                        help="Chọn số lượng giao dịch từ file CSV. Có thể chọn toàn bộ."
                    )
                    
                    skip_rows = st.number_input(
                        "Bỏ qua (rows)",
                        min_value=0,
                        max_value=max(0, len(df) - num_rows),
                        value=0,
                        help="Bỏ qua n dòng đầu để lấy dữ liệu ở vị trí khác"
                    )
                    
                    if st.button(f"📥 Load {num_rows} giao dịch từ CSV", type="primary"):
                        # Show progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Đang đọc file CSV...")
                        progress_bar.progress(30)
                        
                        # Get Amount column, convert to integers
                        amounts = df["Amount"].iloc[skip_rows:skip_rows + num_rows].values
                        
                        status_text.text("Đang xử lý dữ liệu...")
                        progress_bar.progress(70)
                        
                        # Round to integers for easier visualization
                        arr = [int(round(x)) for x in amounts]
                        
                        status_text.text("Hoàn tất!")
                        progress_bar.progress(100)
                        
                        st.session_state["arr"] = arr
                        st.session_state["tree"] = None
                        st.session_state["steps"] = None
                        st.session_state["step_idx"] = 0
                        st.session_state["comparison_results"] = None
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ Đã load {len(arr)} giao dịch từ CSV")
                else:
                    st.error("⚠️ File CSV không có cột 'Amount'")
            except Exception as e:
                st.error(f"❌ Lỗi đọc file CSV: {str(e)}")
        else:
            st.error(f"⚠️ Không tìm thấy file {csv_file}")
        
        # Show description
        st.info(f"ℹ️ {PRESET_DATASETS[dataset_choice]['description']}")
    
    else:
        # Load preset dataset
        preset_data = PRESET_DATASETS[dataset_choice]["data"]
        if st.button(f"✓ Dùng dataset này ({len(preset_data)} phần tử)", type="primary"):
            st.session_state["arr"] = preset_data[:]
            st.session_state["tree"] = None
            st.session_state["steps"] = None
            st.session_state["step_idx"] = 0
            st.session_state["comparison_results"] = None
        
        # Show description
        st.info(f"ℹ️ {PRESET_DATASETS[dataset_choice]['description']}")
    
    # Manual input option
    with st.expander("✏️ Hoặc nhập thủ công"):
        manual = st.text_area(
            "Nhập mảng (cách nhau bởi dấu phẩy)",
            value="81, 14, 3, 94, 35, 31, 28, 17"
        )
        if st.button("✓ Dùng mảng này"):
            try:
                arr = [int(x.strip()) for x in manual.split(",") if x.strip()]
                st.session_state["arr"] = arr
                st.session_state["tree"] = None
                st.session_state["steps"] = None
                st.session_state["step_idx"] = 0
                st.session_state["comparison_results"] = None
            except:
                st.error("Lỗi định dạng! Vui lòng nhập các số cách nhau bởi dấu phẩy.")
    
    st.divider()
    
    # Algorithm selection or comparison
    if page == "🔍 Demo chi tiết thuật toán":
        algo = st.selectbox(
            "Chọn thuật toán",
            [
                "Algorithm 1 - Consecutive Increasing Runs",
                "Algorithm 2 - Consecutive Monotonic Runs", 
                "Algorithm 3 - Non-consecutive L/R"
            ]
        )
        
        if st.button("▶️ Chạy thuật toán", type="primary"):
            arr = st.session_state.get("arr", [])
            if not arr:
                st.error("Vui lòng chọn hoặc nhập mảng trước!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"Đang chạy {algo} với {len(arr)} phần tử...")
                progress_bar.progress(20)
                
                if "Algorithm 1" in algo:
                    tree, steps = build_runs_tree_alg1(arr)
                    st.session_state["algo_type"] = "alg1"
                elif "Algorithm 2" in algo:
                    tree, steps = build_runs_tree_alg2(arr)
                    st.session_state["algo_type"] = "alg2"
                else:
                    tree, steps = build_runs_tree_alg3(arr)
                    st.session_state["algo_type"] = "alg3"
                
                progress_bar.progress(80)
                status_text.text("Đang lưu kết quả...")
                
                st.session_state["tree"] = tree
                st.session_state["steps"] = steps
                st.session_state["sorted"] = tree.array if tree else []
                st.session_state["step_idx"] = 0
                st.session_state["comparison_results"] = None
                
                progress_bar.progress(100)
                status_text.text("Hoàn thành!")
                time.sleep(0.3)
                
                progress_bar.empty()
                status_text.empty()
    
    else:  # Comparison mode
        # Show cache status
        arr = st.session_state.get("arr", [])
        if arr:
            cached_results = load_comparison_cache(arr)
            if cached_results:
                st.success(f"✅ Đã có cache cho {len(arr)} phần tử - Load nhanh!")
                col_run, col_clear = st.columns(2)
                with col_run:
                    use_cache = st.button("📦 Load từ Cache", type="primary")
                with col_clear:
                    if st.button("🗑️ Xóa cache & chạy lại"):
                        clear_cache()
                        cached_results = None
                        st.rerun()
                
                if use_cache:
                    st.session_state["comparison_results"] = cached_results
                    st.session_state["tree"] = None
                    st.session_state["steps"] = None
                    st.session_state["comparison_from_cache"] = True
                    st.rerun()
            else:
                use_cache = False
        
        if st.button("🚀 Chạy so sánh 3 thuật toán", type="primary"):
            arr = st.session_state.get("arr", [])
            if not arr:
                st.error("Vui lòng chọn hoặc nhập mảng trước!")
            else:
                # Run all 3 algorithms and collect results
                results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # ========================================
                # BENCHMARK PHASE: Pure timing (no logging)
                # ========================================
                status_text.text(f"⏱️ Đang benchmark thuần túy (không log) với {len(arr)} phần tử...")
                progress_bar.progress(5)
                
                # Run benchmark multiple times for accurate timing
                benchmark_iterations = 10
                
                # Benchmark Algorithm 1
                start_time = time.time()
                for _ in range(benchmark_iterations):
                    _, runs1, merges1 = benchmark_alg1(arr)
                time1 = (time.time() - start_time) / benchmark_iterations
                
                progress_bar.progress(15)
                
                # Benchmark Algorithm 2
                start_time = time.time()
                for _ in range(benchmark_iterations):
                    _, runs2, merges2 = benchmark_alg2(arr)
                time2 = (time.time() - start_time) / benchmark_iterations
                
                progress_bar.progress(25)
                
                # Benchmark Algorithm 3
                start_time = time.time()
                for _ in range(benchmark_iterations):
                    sorted_arr3, runs3, merges3 = benchmark_alg3(arr)
                time3 = (time.time() - start_time) / benchmark_iterations
                
                progress_bar.progress(35)
                status_text.text("✅ Benchmark hoàn tất! Đang chạy với logging để lấy chi tiết...")
                
                # ========================================
                # LOGGING PHASE: Get steps for visualization
                # ========================================
                
                # Algorithm 1 with logging
                status_text.text(f"🔄 Đang chạy Algorithm 1 với logging...")
                progress_bar.progress(45)
                tree1, steps1 = build_runs_tree_alg1(arr)
                results["Algorithm 1"] = {
                    "tree": tree1,
                    "steps": steps1,
                    "time": time1,  # Use benchmark time!
                    "metrics": calculate_metrics(steps1),
                    "sorted": tree1.array if tree1 else []
                }
                
                progress_bar.progress(55)
                status_text.text(f"✅ Algorithm 1: {time1*1000:.4f}ms (pure), {runs1} runs")
                time.sleep(0.1)
                
                # Algorithm 2 with logging
                status_text.text(f"🔄 Đang chạy Algorithm 2 với logging...")
                progress_bar.progress(65)
                tree2, steps2 = build_runs_tree_alg2(arr)
                results["Algorithm 2"] = {
                    "tree": tree2,
                    "steps": steps2,
                    "time": time2,  # Use benchmark time!
                    "metrics": calculate_metrics(steps2),
                    "sorted": tree2.array if tree2 else []
                }
                
                progress_bar.progress(75)
                status_text.text(f"✅ Algorithm 2: {time2*1000:.4f}ms (pure), {runs2} runs")
                time.sleep(0.1)
                
                # Algorithm 3 with logging
                status_text.text(f"🔄 Đang chạy Algorithm 3 với logging...")
                progress_bar.progress(85)
                tree3, steps3 = build_runs_tree_alg3(arr)
                results["Algorithm 3"] = {
                    "tree": tree3,
                    "steps": steps3,
                    "time": time3,  # Use benchmark time!
                    "metrics": calculate_metrics(steps3),
                    "sorted": tree3.array if tree3 else []
                }
                
                progress_bar.progress(90)
                status_text.text(f"✅ Algorithm 3: {time3*1000:.4f}ms (pure), {runs3} runs")
                time.sleep(0.1)
                
                status_text.text("📊 Đang tổng hợp kết quả...")
                progress_bar.progress(95)
                
                st.session_state["comparison_results"] = results
                st.session_state["tree"] = None
                st.session_state["steps"] = None
                
                # Save to cache for future use
                if len(arr) >= 100:  # Only cache for large datasets
                    status_text.text("💾 Đang lưu cache...")
                    if save_comparison_cache(arr, results):
                        st.session_state["comparison_from_cache"] = False
                
                progress_bar.progress(100)
                time.sleep(0.3)
                
                # Clear progress indicators before rerun
                progress_bar.empty()
                status_text.empty()
                
                # Show success message
                st.success("✅ Hoàn thành so sánh! Xem kết quả bên dưới ⬇️")
                
                # Force rerun to show results
                st.rerun()

# Main content
arr = st.session_state.get("arr", [])
tree = st.session_state.get("tree", None)
steps = st.session_state.get("steps", None)
sorted_arr = st.session_state.get("sorted", None)
comparison_results = st.session_state.get("comparison_results", None)

# ============================================================
# PAGE 1: DEMO CHI TIẾT THUẬT TOÁN
# ============================================================
if page == "🔍 Demo chi tiết thuật toán":
    # Display input array
    st.subheader("📥 Mảng đầu vào")
    if arr:
        # Render as boxes
        html = '<div style="display: flex; flex-wrap: wrap; gap: 2px; margin: 10px 0;">'
        for i, val in enumerate(arr):
            html += f'''<div style="
                border: 1px solid #333;
                background-color: #f0f0f0;
                font-family: monospace;
                min-width: 40px;
                text-align: center;
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    background-color: #ddd;
                    padding: 2px 4px;
                    font-size: 10px;
                    color: #666;
                    border-bottom: 1px solid #333;
                ">{i}</div>
                <div style="
                    padding: 8px 12px;
                    font-size: 14px;
                ">{val}</div>
            </div>'''
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("👈 Hãy tạo mảng ở sidebar")

    # Display result
    st.subheader("📤 Kết quả (sau khi sort)")
    if sorted_arr:
        st.success(f"✓ Đã sắp xếp: {sorted_arr}")
    else:
        st.info("Chưa chạy thuật toán")

    st.divider()

    # Tree visualization and steps
    if tree and steps:
        st.subheader("🌳 Trực quan hóa cây (Tree Visualization)")
        
        total_steps = len(steps)
        step_idx = st.session_state.get("step_idx", 0)
        
        # Navigation
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 3])
        
        with col1:
            if st.button("⏮️ First"):
                st.session_state["step_idx"] = 0
                step_idx = 0
        
        with col2:
            if st.button("⏪ Prev"):
                st.session_state["step_idx"] = max(0, step_idx - 1)
                step_idx = st.session_state["step_idx"]
        
        with col3:
            if st.button("⏩ Next"):
                st.session_state["step_idx"] = min(total_steps - 1, step_idx + 1)
                step_idx = st.session_state["step_idx"]
        
        with col4:
            if st.button("⏭️ Last"):
                st.session_state["step_idx"] = total_steps - 1
                step_idx = total_steps - 1
        
        with col5:
            step_idx = st.slider(
                "Bước",
                0,
                total_steps - 1,
                step_idx,
                format=f"Bước %d / {total_steps}"
            )
            st.session_state["step_idx"] = step_idx
        
        # Display current step info
        if step_idx < total_steps:
            current_step = steps[step_idx]
            st.info(f"**Bước {step_idx + 1}/{total_steps}**: {current_step.get('message', '')}")
            
            # Show step details
            with st.expander("📋 Chi tiết bước này"):
                st.json(current_step)
        
        # Render tree
        algo_type = st.session_state.get("algo_type", "alg1")
        if algo_type == "alg3":
            tree_html = render_tree_html_alg3(step_idx, steps, arr)
        else:
            tree_html = render_tree_html(tree, step_idx, steps, arr)
        st.markdown(tree_html, unsafe_allow_html=True)
        
        st.divider()
        
        # Step list
        st.subheader("📝 Danh sách các bước")
        for i, step in enumerate(steps):
            prefix = "▶️ " if i == step_idx else "• "
            color = "blue" if i == step_idx else "black"
            st.markdown(f"<span style='color: {color};'>{prefix}{i+1}. {step.get('message', '')}</span>", unsafe_allow_html=True)

    else:
        st.info("👈 Chọn thuật toán và bấm 'Chạy thuật toán' để xem visualization")

    st.divider()
    st.caption("💡 Gợi ý: Dùng n = 10-20 để dễ quan sát. Tree sẽ hiển thị quá trình chia và ghép của thuật toán.")

# ============================================================
# PAGE 2: SO SÁNH 3 THUẬT TOÁN
# ============================================================
elif page == "📊 So sánh 3 thuật toán":
    # Display input array
    st.subheader("📥 Mảng đầu vào")
    if arr:
        st.write(f"Mảng: {arr[:20]}{'...' if len(arr) > 20 else ''}")
        st.write(f"Số phần tử: {len(arr)}")
    else:
        st.info("👈 Hãy chọn dataset ở sidebar")
    
    st.divider()
    
    # Comparison results
    if comparison_results:
        st.header("📊 So sánh hiệu suất 3 thuật toán")
        
        # Create comparison table
        st.subheader("📈 Bảng so sánh metrics")
        
        # Prepare data for table
        algo_names = list(comparison_results.keys())
        
        # Display metrics in columns
        cols = st.columns(3)
        for idx, (algo_name, result) in enumerate(comparison_results.items()):
            with cols[idx]:
                metrics = result["metrics"]
                st.markdown(f"### {algo_name}")
                st.metric("Thời gian", f"{result['time']*1000:.2f} ms")
                st.metric("Số runs phát hiện", metrics["num_runs"])
                st.metric("Số lần merge", metrics["num_merges"])
                st.metric("Tổng số bước", metrics["total_steps"])
                
                if metrics["num_runs"] > 0:
                    st.metric("Độ dài run trung bình", f"{metrics['avg_run_length']:.2f}")
                    st.metric("Run dài nhất", metrics["max_run_length"])
                    st.metric("Run ngắn nhất", metrics["min_run_length"])
                
                if algo_name == "Algorithm 3":
                    st.metric("Số lần flush block", metrics["num_flushes"])
        
        st.divider()
        
        # Detailed comparison charts
        st.subheader("📊 Biểu đồ so sánh")
        
        # Chart 1: Execution time
        st.markdown("#### ⏱️ Thời gian thực thi")
        chart_data = {
            "Thuật toán": algo_names,
            "Thời gian (ms)": [comparison_results[algo]["time"] * 1000 for algo in algo_names]
        }
        st.bar_chart(chart_data, x="Thuật toán", y="Thời gian (ms)")
        
        # Chart 2: Number of runs
        st.markdown("#### 🔢 Số lượng runs phát hiện được")
        chart_data = {
            "Thuật toán": algo_names,
            "Số runs": [comparison_results[algo]["metrics"]["num_runs"] for algo in algo_names]
        }
        st.bar_chart(chart_data, x="Thuật toán", y="Số runs")
        
        # Chart 3: Number of merges
        st.markdown("#### 🔀 Số lần merge")
        chart_data = {
            "Thuật toán": algo_names,
            "Số merges": [comparison_results[algo]["metrics"]["num_merges"] for algo in algo_names]
        }
        st.bar_chart(chart_data, x="Thuật toán", y="Số merges")
        
        # Chart 4: Total steps
        st.markdown("#### 📝 Tổng số bước thực thi")
        chart_data = {
            "Thuật toán": algo_names,
            "Số bước": [comparison_results[algo]["metrics"]["total_steps"] for algo in algo_names]
        }
        st.bar_chart(chart_data, x="Thuật toán", y="Số bước")
        
        # Analysis and recommendations
        st.divider()
        st.subheader("💡 Phân tích & Khuyến nghị")
        
        # Find best algorithm for different metrics
        best_time = min(algo_names, key=lambda x: comparison_results[x]["time"])
        best_runs = min(algo_names, key=lambda x: comparison_results[x]["metrics"]["num_runs"])
        best_merges = min(algo_names, key=lambda x: comparison_results[x]["metrics"]["num_merges"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"🏆 **Nhanh nhất**: {best_time}")
        with col2:
            st.success(f"🏆 **Ít runs nhất**: {best_runs}")
        with col3:
            st.success(f"🏆 **Ít merges nhất**: {best_merges}")
        
        # Detailed analysis
        with st.expander("📖 Xem phân tích chi tiết"):
            st.markdown("""
            ### Giải thích kết quả:
            
            **Algorithm 1 (Consecutive Increasing Runs)**:
            - Phù hợp với dữ liệu có nhiều dãy con tăng dần liên tiếp
            - Ví dụ: Dữ liệu từ nhiều cảm biến IoT, mỗi cảm biến đo giá trị tăng dần
            
            **Algorithm 2 (Consecutive Monotonic Runs)**:
            - Phù hợp với dữ liệu có cả dãy tăng và giảm xen kẽ
            - Ví dụ: Lịch sử giao dịch ngân hàng (nạp/rút tiền), giá cổ phiếu
            
            **Algorithm 3 (Non-consecutive L/R)**:
            - Phù hợp với dữ liệu xen kẽ không theo pattern cố định
            - Ví dụ: Điểm thi xen kẽ môn dễ/khó, dữ liệu thời tiết không đều
            
            **Metrics quan trọng**:
            - **Số runs**: Càng ít càng tốt (ít phải merge hơn)
            - **Số merges**: Phản ánh độ phức tạp thuật toán
            - **Thời gian**: Hiệu suất thực tế trên máy
            """)
    else:
        st.info("👈 Chọn dataset và bấm 'Chạy so sánh 3 thuật toán' ở sidebar")

