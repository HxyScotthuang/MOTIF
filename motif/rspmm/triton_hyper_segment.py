import torch

import triton
import triton.language as tl


# Hypergraph (relation graph) sum aggregation with an edge-parallel, segmented-reduction layout.
#
# Semantics are identical to HyperRelConvSumAggr in triton_rspmm.py:
#   out[i] = sum_{incidences k at destination i} rels[type_k] * prod_q (h[src_qk] + pos[pos_qk])
# where a source equal to 0 is the padding node and contributes a factor of 1.
#
# The relation graph has very few nodes (~2|R|) but millions of hyperedges and only a handful of
# hyperedge types. The node-parallel kernels in triton_rspmm.py therefore launch only ~|V| programs that
# each walk tens of thousands of incidences serially, and the backward pass funnels every incidence into
# |types| rows of drel with atomics. Here, incidences are sorted by a segment key and split into fixed
# size chunks; each program reduces its chunk in registers and only issues an atomic when the segment
# changes. This gives |incidences| / CHUNK programs regardless of the node count, and no per-incidence
# atomics.
#
# Since the graph is small, all factors are gathered from two small dense tables:
#   hp[node * P + pos]   = h[node] + pos_encodings[pos]   (1 for the padding node), P = max_arity + 1
#   gr[dst * T + type]   = dh_out[dst] * rels[type]
# and the forward pass keeps seg[dst * T + type] = sum_k prod_q hp[...], so that
#   out = sum_t seg[:, t] * rels[t]      and      drel = sum_i seg[i, :] * dh_out[i]
# are dense contractions. The backward pass computes the "product of the other factors" directly instead
# of dividing the full product by the own factor, so it is exact also when a factor is (close to) zero.


@triton.jit
def hyper_segment_sum_kernel_forward(
      seg_out,       # output [num_nodes * NUM_TYPES, IN_CHAN], zero initialised
      seg_ids,       # segment (dst * NUM_TYPES + type) of every incidence, sorted, shape [num_inc]
      hp_index,      # row of hp for every source, shape [NUM_SRC, num_inc]
      hp,            # factor table [num_nodes * P, IN_CHAN]
      NUM_INC,       # number of incidences
      IN_CHAN        : tl.constexpr,
      NUM_SRC        : tl.constexpr, # max_arity - 1
      CHUNK          : tl.constexpr, # incidences per program
      WG_SIZE        : tl.constexpr, # features per program
    ):
    # seg_out[s] = sum_{k in segment s} prod_q hp[hp_index[q, k]]
    start = tl.program_id(0) * CHUNK
    end = tl.minimum(start + CHUNK, NUM_INC)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN

    cur_seg = tl.load(seg_ids + start)
    acc = tl.zeros((WG_SIZE,), dtype=tl.float32)
    for k in range(start, end):
        seg = tl.load(seg_ids + k)
        if seg != cur_seg:
            tl.atomic_add(seg_out + cur_seg * IN_CHAN + feat_offsets, acc, feat_valid_mask)
            acc = tl.zeros((WG_SIZE,), dtype=tl.float32)
            cur_seg = seg
        prod = tl.full((WG_SIZE,), 1, dtype=tl.float32)
        for q in tl.static_range(NUM_SRC):
            row = tl.load(hp_index + q * NUM_INC + k)
            prod *= tl.load(hp + row * IN_CHAN + feat_offsets, feat_valid_mask, 0.0)
        acc += prod
    tl.atomic_add(seg_out + cur_seg * IN_CHAN + feat_offsets, acc, feat_valid_mask)


@triton.jit
def hyper_segment_sum_kernel_backward(
      dhp,           # output [num_nodes * P, IN_CHAN], zero initialised
      grad_rows,     # hp row of every (incidence, place) pair with a non-padding source, sorted, shape [num_pairs]
      grad_inc,      # incidence of every pair, shape [num_pairs]
      grad_place,    # place q of the source within its incidence, shape [num_pairs]
      seg_ids,       # segment (dst * NUM_TYPES + type) of every incidence, shape [num_inc]
      hp_index,      # row of hp for every source, shape [NUM_SRC, num_inc]
      hp,            # factor table [num_nodes * P, IN_CHAN]
      gr,            # gradient table [num_nodes * NUM_TYPES, IN_CHAN]
      NUM_INC,       # number of incidences
      NUM_PAIRS,     # number of pairs
      IN_CHAN        : tl.constexpr,
      NUM_SRC        : tl.constexpr,
      CHUNK          : tl.constexpr,
      WG_SIZE        : tl.constexpr,
    ):
    # dhp[r] = sum_{pairs (k, q) with hp_index[q, k] = r} gr[seg_k] * prod_{q' != q} hp[hp_index[q', k]]
    start = tl.program_id(0) * CHUNK
    end = tl.minimum(start + CHUNK, NUM_PAIRS)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN

    cur_row = tl.load(grad_rows + start)
    acc = tl.zeros((WG_SIZE,), dtype=tl.float32)
    for i in range(start, end):
        grad_row = tl.load(grad_rows + i)
        if grad_row != cur_row:
            tl.atomic_add(dhp + cur_row * IN_CHAN + feat_offsets, acc, feat_valid_mask)
            acc = tl.zeros((WG_SIZE,), dtype=tl.float32)
            cur_row = grad_row
        k = tl.load(grad_inc + i)
        place = tl.load(grad_place + i)
        seg = tl.load(seg_ids + k)
        prod = tl.load(gr + seg * IN_CHAN + feat_offsets, feat_valid_mask, 0.0)
        for q in tl.static_range(NUM_SRC):
            if place != q:
                row = tl.load(hp_index + q * NUM_INC + k)
                prod *= tl.load(hp + row * IN_CHAN + feat_offsets, feat_valid_mask, 0.0)
        acc += prod
    tl.atomic_add(dhp + cur_row * IN_CHAN + feat_offsets, acc, feat_valid_mask)


# The sorted incidence layout depends only on the graph, so it is built once and reused by all layers
# (and by the backward pass) as long as the same CSR tensors are passed in.
_plan_cache = {}


def build_hyper_segment_plan(rowptr, indices, edge_types, pos_index, num_types, num_pos):
    cached = _plan_cache.get("plan")
    # the cache keeps references to its key tensors, so their memory cannot be reused while cached
    if cached is not None and cached[0] is indices and cached[1] is rowptr and cached[2] == (num_types, num_pos):
        return cached[3]

    num_nodes = rowptr.numel() - 1
    num_src, num_inc = indices.shape
    device = indices.device
    inc_dst = torch.repeat_interleave(torch.arange(num_nodes, device=device), rowptr[1:] - rowptr[:-1],
                                      output_size=num_inc)
    # sort incidences by (destination, type) so that each (dst, type) segment is contiguous
    seg_ids, perm = torch.sort(inc_dst * num_types + edge_types, stable=True)
    hp_index = (indices[:, perm] * num_pos + pos_index[:, perm]).contiguous()
    # (incidence, place) pairs with a non-padding source (rows >= num_pos), sorted by hp row
    grad_rows = hp_index.flatten()
    pair = torch.arange(num_src * num_inc, device=device)
    valid = grad_rows >= num_pos
    grad_rows, pair = grad_rows[valid], pair[valid]
    grad_rows, order = torch.sort(grad_rows, stable=True)
    pair = pair[order]
    plan = dict(seg_ids=seg_ids, hp_index=hp_index, grad_rows=grad_rows,
                grad_inc=(pair % num_inc).contiguous(), grad_place=(pair // num_inc).contiguous())
    _plan_cache["plan"] = (indices, rowptr, (num_types, num_pos), plan)
    return plan


class HyperRelConvSumAggrSegmented(torch.autograd.Function):
    # Drop-in replacement for triton_rspmm.HyperRelConvSumAggr (same arguments, same gradients returned).
    CHUNK = 256
    WG_SIZE = 512
    NUM_WARPS = 4

    @staticmethod
    def forward(ctx, h_in, rowptr, indices, out_node_count, edge_types, rel_features, pos_encodings, pos_index, work_group_size = None):
        torch.cuda.set_device(h_in.device)
        num_features = h_in.shape[1]
        # Like the original kernel, rel_features is read as a row-major [num_types, num_features] buffer
        # (layers.py passes a transposed view of such a buffer).
        num_types = rel_features.numel() // num_features
        num_pos = pos_encodings.shape[0]
        rels = rel_features.as_strided((num_types, num_features), (num_features, 1))
        plan = build_hyper_segment_plan(rowptr, indices, edge_types, pos_index, num_types, num_pos)

        hp = (h_in.unsqueeze(1) + pos_encodings.unsqueeze(0)).view(-1, num_features)
        hp[:num_pos] = 1  # padding node
        chunk, wg_size = HyperRelConvSumAggrSegmented.CHUNK, HyperRelConvSumAggrSegmented.WG_SIZE
        num_inc = plan["seg_ids"].numel()
        seg_out = torch.zeros((out_node_count * num_types, num_features), dtype=h_in.dtype, device=h_in.device)
        if num_inc > 0:
            grid = (triton.cdiv(num_inc, chunk), triton.cdiv(num_features, wg_size))
            hyper_segment_sum_kernel_forward[grid](
                seg_out, plan["seg_ids"], plan["hp_index"], hp, num_inc,
                num_features, plan["hp_index"].shape[0], chunk, wg_size, num_warps=HyperRelConvSumAggrSegmented.NUM_WARPS)
        seg_out = seg_out.view(out_node_count, num_types, num_features)
        h_out = (seg_out * rels.unsqueeze(0)).sum(1)

        ctx.save_for_backward(rel_features, hp, seg_out)
        ctx.plan = plan
        return h_out

    @staticmethod
    def backward(ctx, h_out_grad):
        rel_features, hp, seg_out = ctx.saved_tensors
        plan = ctx.plan
        num_nodes, num_types, num_features = seg_out.shape
        rels = rel_features.as_strided((num_types, num_features), (num_features, 1))
        h_out_grad = h_out_grad.contiguous()

        # drel[t] = sum_i dh_out[i] * seg_out[i, t]
        rel_in_grad = torch.empty_like(rel_features)
        rel_in_grad.as_strided((num_types, num_features), (num_features, 1)).copy_(
            (seg_out * h_out_grad.unsqueeze(1)).sum(0))

        gr = (h_out_grad.unsqueeze(1) * rels.unsqueeze(0)).view(-1, num_features)
        chunk, wg_size = HyperRelConvSumAggrSegmented.CHUNK, HyperRelConvSumAggrSegmented.WG_SIZE
        dhp = torch.zeros_like(hp)
        num_pairs = plan["grad_rows"].numel()
        if num_pairs > 0:
            grid = (triton.cdiv(num_pairs, chunk), triton.cdiv(num_features, wg_size))
            hyper_segment_sum_kernel_backward[grid](
                dhp, plan["grad_rows"], plan["grad_inc"], plan["grad_place"], plan["seg_ids"], plan["hp_index"],
                hp, gr, plan["seg_ids"].numel(), num_pairs,
                num_features, plan["hp_index"].shape[0], chunk, wg_size, num_warps=HyperRelConvSumAggrSegmented.NUM_WARPS)
        # gradients w.r.t. the positional encodings are not propagated, as in the original kernel
        h_in_grad = dhp.view(num_nodes, -1, num_features).sum(1)
        return h_in_grad, None, None, None, None, rel_in_grad, None, None, None
