from dataclasses import replace
import imp
import gs
from sys import meta_path
import torch
import load_graph
import dgl
from dgl.transforms.functional import to_block
from dgl.sampling import random_walk, pack_traces
import time
import numpy as np
import argparse

seed_tensor = torch.LongTensor([[173816, 208829,  63362, 116932, 205017],
                                [45415, 207539, 203974,   5114, 197892],
                                [71024, 131417,  74116, 153943,  79153],
                                [226324,  74649,  74214, 201846, 136278],
                                [171820,  97903, 230369,  48546,  18152],
                                [212355, 199777, 190252, 168841, 103338],
                                [83853,   9057,  57328,  85878, 115405],
                                [209479,   4580,  88659, 194252,    472],
                                [72574, 180745, 124847, 202254, 178141],
                                [165461,  33359,  26576,  68844, 179626],
                                [163810,  60995, 147504, 110843, 190783],
                                [177007, 225389, 134003,  54107, 194030],
                                [56565,  78101, 205247, 126099,  27090],
                                [232520,  26872, 139485,  96780,  78190],
                                [26294,  14682,  44901, 158220,  33535],
                                [116002, 143548, 229932, 135258, 203766],
                                [182698, 221008, 118542,   3354,  47717],
                                [141526, 176680,  39667, 129522,   7493],
                                [35192,  36779, 224653, 172145,  74456],
                                [178467, 110607, 163831, 188066,  59294],
                                [60401,  25071,  74831, 162113, 168376],
                                [47587, 202474,  73669, 120143,  67745],
                                [100315,  47844, 155170, 141960, 182117],
                                [147339,  32445, 189532, 184151,  15324],
                                [13285, 100992,  98446, 137928,   9163],
                                [141912,   5375, 197301, 105081, 114291],
                                [134117, 134325, 119073,  73736,  45609],
                                [185492,  89250, 184134, 106456, 106849],
                                [156286,  56425, 188723, 103310,  22160],
                                [35012,  27080, 134601, 100771, 161408],
                                [96183,  47163,  27044, 186347, 229140],
                                [152719, 212772,  96271, 108007, 177514],
                                [7091, 113133,  45374, 102088,  61151],
                                [166242, 141315, 180331, 230055, 139848],
                                [209820, 101503,    229,  38839,  16430],
                                [12673,  75970, 176218, 168242, 228033],
                                [105136, 115519, 158584,  61086,  46963],
                                [17924, 218503, 113628, 160192,  55697],
                                [222293,  75673, 208536, 150915, 221358],
                                [38212, 174667, 127341,  94609,  29510],
                                [52586, 230753,  47497,  14132,  70350],
                                [183623,  92906,   7662, 123587, 104641],
                                [35190,  19003, 207764,  57886,  37354],
                                [147570, 189844,  80020,  64755,  50052],
                                [198829,  89845, 118029,  96747,  62088],
                                [146952,  63755, 143205,  54415,  99131],
                                [213436, 148342,  31856,  15212, 184478],
                                [103898, 221836,  42923, 111480, 211466],
                                [16418,   8826,    617,  59693, 131195],
                                [187721, 103407, 167160, 227011,  66620],
                                [9164,  16872, 186980, 146465, 180470],
                                [83301, 169785,  73560, 223502, 208813],
                                [218465, 227988, 168670, 219139, 106838],
                                [176043,  14795,  11606,   1255,  77461],
                                [32554,   6378,  96666, 105837,    948],
                                [880, 123393,  67794, 134458, 220969],
                                [27007,  80916, 224537,  53897,  87267],
                                [4065, 149045,   1691,  52529, 210152],
                                [34744, 186747, 152679,  42181,  53507],
                                [149259,  17013, 202857,  30279, 198201],
                                [126409, 194222,  13958,  88847,  24430],
                                [143488, 125519, 227399, 203450, 109068],
                                [53571,  50549, 122850,  10287,  97855],
                                [66140, 131996,  33737,   8603,  17473],
                                [191954, 172167, 198835, 189420, 144913],
                                [212365,  14026, 119135, 224618, 120492],
                                [139837,  98904, 130949,  47129, 125224],
                                [75358, 163363, 147685, 151399, 181876],
                                [133912,  67902,  92305, 184801,  15663],
                                [45253,  50718, 134425, 147218, 144784],
                                [187566, 118761, 182516,  48964, 107468],
                                [55044, 118211,  33120, 200628, 106874],
                                [193811,  77454, 163250, 171971, 196530],
                                [29369, 166290,  31876, 169357, 214924],
                                [221906,  33480, 104359,  31381, 156254],
                                [183967, 219393,  87940, 127416,  69603],
                                [63938, 107994,  42746, 151563, 132774],
                                [31740, 123386, 107687, 140846,  45840],
                                [107037,  22625, 117875,  76850, 173015],
                                [178006, 151489,   7989, 126763, 126432],
                                [66521, 216970,  33833, 171498,  79689],
                                [180301, 100577, 171762,  54319,  53841],
                                [213601,  85025,   1552,  20127,  11279],
                                [187721,  17212,  90243, 220784, 140468],
                                [213244, 114384, 227095, 118387, 203554],
                                [172634, 154516, 215838, 120023,  89213],
                                [104915,  49557, 125726, 134562,   1878],
                                [15928, 124387, 158270, 216461, 196748],
                                [178523, 135742, 228417, 159378,  87507],
                                [205771, 108326,  13670,  65967,  24169],
                                [187147, 163720,   3447,  52516, 202110],
                                [120360, 146781, 108847, 154601, 217618],
                                [187926, 178938, 211124, 128633,    575],
                                [92536,  65042, 205529, 136477,  83541],
                                [193929,  92117, 232348, 173113, 187405],
                                [206902,  41769,  68420, 227165, 109856],
                                [167013,  93721, 151302, 132364,  12825],
                                [154246, 116597, 214296, 106721, 219282],
                                [131185,  45089, 124675, 160930,  99414],
                                [122748,  92217, 220371, 109484, 110905]]).to('cuda')


def main(args):
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    dgl_g = dgl_graph.long()
    dgl_g = dgl_g.to("cuda")
    matrix = gs.Matrix(gs.Graph(False))
    matrix.load_dgl_graph(dgl_graph)

    def shadowgnn_dgl(graph: dgl.DGLGraph, seed_nodes, fanouts):
        output_nodes = seed_nodes
        for fanout in reversed(fanouts):
            torch.cuda.nvtx.range_push("shadowgnn dgl sample neighbors")
            frontier = graph.sample_neighbors(seed_nodes, fanout)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("shadowgnn dgl to block")
            block = dgl.transforms.to_block(frontier, seed_nodes)
            torch.cuda.nvtx.range_pop()
            seed_nodes = block.srcdata[dgl.NID]
        torch.cuda.nvtx.range_push("shadowgnn dgl induce subgraph")
        subg = graph.subgraph(seed_nodes, relabel_nodes=True)
        torch.cuda.nvtx.range_pop()
        return seed_nodes, output_nodes, subg

    def shadowgnn_nonfused(A: gs.Matrix, seeds, fanouts):
        output_nodes = seeds
        for fanout in reversed(fanouts):
            torch.cuda.nvtx.range_push(
                "shadowgnn columnwise slicing and sampling")
            subA = A[:, seeds]
            subA = subA.columnwise_sampling(fanout, False)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("shadowgnn all_indices")
            seeds = subA.all_indices()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("shadowgnn induce subgraph")
        retA = A[seeds, seeds]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("shadowgnn subgraph relabel")
        retA = retA.relabel()
        torch.cuda.nvtx.range_pop()
        return seeds, output_nodes, retA

    def shadowgnn_fused(A: gs.Matrix, seeds, fanouts):
        output_nodes = seeds
        for fanout in reversed(fanouts):
            torch.cuda.nvtx.range_push(
                "shadowgnn columnwise slicing and sampling")
            subA = A.fused_columnwise_slicing_sampling(seeds, fanout, False)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("shadowgnn all_indices")
            seeds = subA.all_indices()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("shadowgnn induce subgraph")
        retA = A[seeds, seeds]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("shadowgnn subgraph relabel")
        retA = retA.relabel()
        torch.cuda.nvtx.range_pop()
        return seeds, output_nodes, retA

    def bench(loop_num, func, args):
        time_list = []

        for i in range(loop_num):
            torch.cuda.synchronize()
            begin = time.time()
            seed_nodes, output_nodes, subg = func(
                args[0], seed_tensor[i, :], args[2])
            #print("ret nodes:", seed_nodes.numel())
            torch.cuda.synchronize()
            end = time.time()
            time_list.append(end - begin)
        print(func.__name__, " AVG:", np.mean(time_list[10:]) * 1000, " ms.")
    fanouts = [5, 15, 25]
    seeds = torch.randint(0,  232965, (5, 100), device='cuda')
    if args.sample_alg == "dgl":
        bench(args.loops,
              shadowgnn_dgl, args=(
                  dgl_g,
                  seeds,
                  fanouts))
    elif args.sample_alg == "nonfused":
        bench(args.loops,
              shadowgnn_nonfused, args=(
                  matrix,
                  seeds,
                  fanouts))
    elif args.sample_alg == "fused":
        bench(args.loops,
              shadowgnn_fused, args=(
                  matrix,
                  seeds,
                  fanouts))
    else:
        bench(args.loops,
              shadowgnn_dgl, args=(
                  dgl_g,
                  seeds,
                  fanouts))
        bench(args.loops,
              shadowgnn_nonfused, args=(
                  matrix,
                  seeds,
                  fanouts))
        bench(args.loops,
              shadowgnn_fused, args=(
                  matrix,
                  seeds,
                  fanouts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAINT')
    # sampler params
    parser.add_argument("--sample_alg", type=str, default="all", choices=['dgl', 'nonfused', 'fused', 'all'],
                        help="Type of sample algorithm")
    # training params
    parser.add_argument("--loops", type=int, default=100,
                        help="Number of test loops")
    parser.add_argument("--seed_num", type=int, default=5,
                        help="Number of seed num")
    args = parser.parse_args()
    main(args)
