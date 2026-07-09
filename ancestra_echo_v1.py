import os
import sys
import argparse
import logging
import json
import copy
from datetime import datetime
from pathlib import Path
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import BiopythonWarning
import warnings
from ete3 import Tree
from matplotlib import cm
import mplcursors
from functools import lru_cache
from scipy.stats import gaussian_kde
import networkx as nx
import pandas as pd

warnings.filterwarnings("ignore", message="Attempting to set identical low and high xlims*")
warnings.simplefilter('ignore', BiopythonWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IUPAC_CODES = {
    'A': {'a'}, 'C': {'c'}, 'G': {'g'}, 'T': {'t'},
    'R': {'a', 'g'}, 'Y': {'c', 't'}, 'W': {'a', 't'},
    'S': {'g', 'c'}, 'K': {'g', 't'}, 'M': {'a', 'c'},
    'B': {'c', 'g', 't'}, 'D': {'a', 'g', 't'}, 'H': {'a', 'c', 't'},
    'V': {'a', 'c', 'g'}, 'N': {'a', 'c', 'g', 't'}
}

NUC_ALPHABET  = ['a', 'c', 'g', 't']
AA_ALPHABET   = list('ACDEFGHIKLMNPQRSTVWY')   # 20 standard amino acids
PSEUDOCOUNT   = 0.5                             # Laplace-like pseudocount

REGION_FIELDS = [
    "cdr1_start", "cdr1_end",
    "cdr2_start", "cdr2_end",
    "cdr3_start", "cdr3_end",
    "fwr1_start", "fwr1_end",
    "fwr2_start", "fwr2_end",
    "fwr3_start", "fwr3_end",
    "fwr4_start", "fwr4_end",
]
# ---------------------------------------------------------------------------
# PSSM class
# ---------------------------------------------------------------------------

class PSSM:
    """
    Position-Specific Scoring Matrix built from a set of aligned sequences X.

    Parameters
    ----------
    sequences : list[str]
        Aligned sequences (all same length).  Already in the target alphabet
        (nucleotide or amino acid).
    alphabet : list[str]
        Letters that can appear (lowercase for nucleotide, uppercase for AA).
    scoring : str
        'logodds'   – log( f_i / b_i ) summed across positions, then normalised.
        'frequency' – product of positional probabilities (summed in log space),
                      then normalised.
    background : str
        'uniform'   – equal weight for every letter in the alphabet.
        'observed'  – frequencies computed from the whole alignment X.
    """

    def __init__(self, sequences, alphabet, regions_mut_rate, scoring='logodds', background='uniform'):
        self.alphabet  = alphabet
        self.alpha_idx = {c: i for i, c in enumerate(alphabet)}
        self.regions_mu = regions_mut_rate
        self.scoring   = scoring
        self.background = background
        self.width     = len(sequences[0])
        self.n_seqs    = len(sequences)

        self._count_matrix  = self._build_count_matrix(sequences)
        self._freq_matrix   = self._build_freq_matrix()
        self._bg_freq       = self._build_background(sequences)
        self._score_matrix  = self._build_score_matrix()

        # Pre-compute min/max achievable scores for normalisation
        self._min_score, self._max_score = self._compute_score_range()
        self.inseq_scores = {seq: self.score(seq) for seq in sequences}
        self.kde = gaussian_kde(list(self.inseq_scores.values()))

        logger.info(
            f"PSSM built | width={self.width} | seqs={self.n_seqs} | "
            f"alphabet={'nuc' if len(alphabet)==4 else 'aa'} | "
            f"scoring={scoring} | background={background}"
        )

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_count_matrix(self, sequences):
        """Raw counts: shape (width, |alphabet|)."""
        n = len(self.alphabet)
        counts = np.zeros((self.width, n), dtype=float)
        for seq in sequences:
            seq = seq.lower() if self.alphabet == NUC_ALPHABET else seq.upper()
            for pos, char in enumerate(seq):
                if pos >= self.width:
                    break
                if char in self.alpha_idx:
                    counts[pos, self.alpha_idx[char]] += 1.0
        return counts

    def _build_freq_matrix(self):
        """Smoothed frequency matrix with pseudocount."""
        n = len(self.alphabet)
        smoothed = self._count_matrix + PSEUDOCOUNT
        row_sums  = smoothed.sum(axis=1, keepdims=True)
        return smoothed / row_sums          # shape (width, |alphabet|)

    def _build_background(self, sequences):
        """Background frequency vector, shape (|alphabet|,)."""
        n = len(self.alphabet)
        if self.background == 'uniform':
            return np.ones(n) / n
        else:   # 'observed'
            total = self._count_matrix.sum(axis=0) + PSEUDOCOUNT
            return total / total.sum()

    def _build_score_matrix(self):
        """
        Score matrix: shape (width, |alphabet|).
        logodds  → log2( freq / bg )
        frequency → log2( freq )          (log-probability, avoids underflow)
        """
        if self.scoring == 'logodds':
            # Avoid division by zero (bg is already smoothed via pseudocount)
            bg = np.where(self._bg_freq == 0, 1e-9, self._bg_freq)
            return np.log2(self._freq_matrix / bg)
        else:   # 'frequency'
            freq = np.where(self._freq_matrix == 0, 1e-9, self._freq_matrix)
            return np.log2(freq)

    def _compute_score_range(self):
        """Min and max possible scores over the full sequence length."""
        min_score = self._score_matrix.min(axis=1).sum()
        max_score = self._score_matrix.max(axis=1).sum()
        return min_score, max_score

    # ------------------------------------------------------------------
    # Public scoring
    # ------------------------------------------------------------------

    def score(self, seq):
        """
        Score *seq* against the PSSM.  Returns a float in [0, 1].

        If seq is shorter than PSSM width, only the overlapping region is
        scored and the score is normalised against that region's range.
        If seq is longer, the best-scoring window of PSSM width is used.
        """
        seq = seq.lower() if self.alphabet == NUC_ALPHABET else seq.upper()
        seq_len = len(seq)

        if seq_len == 0:
            return 0.0

        if seq_len >= self.width:
            # Slide a window of PSSM width and take the best score
            raw = self._best_window_score(seq)
            lo, hi = self._min_score, self._max_score
        else:
            # Score the overlap only
            raw = self._partial_score(seq)
            lo = self._score_matrix[:seq_len].min(axis=1).sum()
            hi = self._score_matrix[:seq_len].max(axis=1).sum()

        if hi == lo:
            return 1.0
        normalised = (raw - lo) / (hi - lo)
        return float(max(0.0, min(1.0, normalised)))

    def _best_window_score(self, seq):
        best = -np.inf
        n_windows = len(seq) - self.width + 1
        for start in range(n_windows):
            window = seq[start: start + self.width]
            s = self._window_score(window)
            if s > best:
                best = s
        return best

    # def _window_score(self, window):
    #     total = 0.0
    #     for pos, char in enumerate(window):
    #         idx = self.alpha_idx.get(char)
    #         if idx is not None:
    #             total += self._score_matrix[pos, idx]
    #         else:
    #             # Unknown character gets the minimum score at this position
    #             total += self._score_matrix[pos].min()
    #     return total

    # def _partial_score(self, seq):
    #     total = 0.0
    #     for pos, char in enumerate(seq):
    #         idx = self.alpha_idx.get(char)
    #         if idx is not None:
    #             total += self._score_matrix[pos, idx]
    #         else:
    #             total += self._score_matrix[pos].min()
    #     return total

    def _window_score(self, window):
        indices = np.array([self.alpha_idx.get(c, -1) for c in window])
        valid = indices >= 0
        row_idx = np.arange(self.width)
        scores = np.where(
            valid,
            self._score_matrix[row_idx, np.where(valid, indices, 0)],
            self._score_matrix.min(axis=1)
        )
        return float(scores.sum())

    def _partial_score(self, seq):
        seq_len = len(seq)
        indices = np.array([self.alpha_idx.get(c, -1) for c in seq])
        valid = indices >= 0
        row_idx = np.arange(seq_len)
        scores = np.where(
            valid,
            self._score_matrix[:seq_len][row_idx, np.where(valid, indices, 0)],
            self._score_matrix[:seq_len].min(axis=1)
        )
        return float(scores.sum())
# ---------------------------------------------------------------------------
# Cached translation helper
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def translate_nucleotide_to_protein_min_stops_cached(nuc_seq):
    return translate_nucleotide_to_protein_min_stops(nuc_seq)


# ---------------------------------------------------------------------------
# BCRSimulator
# ---------------------------------------------------------------------------

class BCRSimulator:
    """
    Generates BCR repertoires via somatic hypermutation starting from a
    given naive (root) sequence, selecting on PSSM fitness instead of
    antigen affinity.
    """

    def __init__(self):
        self.bcr_counter = 1

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _generate_bcr_id(self):
        bcr_id = f"seq{self.bcr_counter}"
        self.bcr_counter += 1
        return bcr_id

    # ------------------------------------------------------------------
    # Root BCR from naive sequence
    # ------------------------------------------------------------------

    def build_root_bcr(self, naive_nuc_sequence):
        """
        Build the root BCR dict directly from the given naive nucleotide
        sequence.  No VDJ recombination is performed.
        """
        naive_nuc_sequence = naive_nuc_sequence.lower().replace('u', 't')
        a_sequence, min_stop, best_frame = translate_nucleotide_to_protein_min_stops(naive_nuc_sequence)
        return {
            "id":         "naive",
            "sequence":   naive_nuc_sequence,
            "a_sequence": a_sequence,
            "frame":      best_frame,
            "generation": 0,
            "parent":     None,
            "mutations":  0,
            "affinity":   0.0,   # will be filled in by caller
            "abundance":  0,
        }

    # ------------------------------------------------------------------
    # Mutation engine  (unchanged from original)
    # ------------------------------------------------------------------

    def _mutate_sequence(self, sequence, regions_mutation_rates):
        nucleotides = list(sequence)
        mutations   = 0
        VALID_BASES = {'a', 'c', 'g', 't'}
        seq_len = len(nucleotides)
        region_of = [None] * seq_len
        for region, info in regions_mutation_rates.items():
            if region =='none_region':
                for pos in info['positions']:
                    region_of[pos] = region
            else:
                for pos in range(info['start'], info['end']):
                    region_of[pos] = region

        for i, base in enumerate(nucleotides):
            region = region_of[i]
            rates_info = regions_mutation_rates[region]
            if random.random() >= rates_info["p_rate"]:
                continue

            if base in VALID_BASES:
                nucleotides[i] = biased_mutation(base, rates_info["p_ti"])
                mutations += 1

        mutated_sequence = ''.join(nucleotides)
        if mutated_sequence == sequence:
            mutations = 0
        return mutated_sequence, mutations

    # ------------------------------------------------------------------
    # Repertoire generation
    # ------------------------------------------------------------------

    def generate_repertoire(self, pssm, pssm_mode, max_generations, minimum_num_unique_sequences, naive_nuc_sequence):

        root = self.build_root_bcr(naive_nuc_sequence)
        root["affinity"] = self._pssm_score(pssm, pssm_mode, root)
        warmup_path=[]
        thresh = np.min(pssm.kde.dataset)

        if root["affinity"] < thresh:
            print('The germline score being sub-threshold, the warm-up routine is now active.')
            MAX_WARMUP_STEPS = 10000
            current_seq   = root["sequence"]
            current_a_seq = root["a_sequence"]
            current_frame = root["frame"]
            total_warmup_mutations = 0
            warmup_regions_mu = {
                region: {**info, "p_rate": 0.001, "p_ti": 0.7, "p_tv": 0.3}
                for region, info in pssm.regions_mu.items()
            }
            warmup_thresh = np.mean(pssm.kde.dataset)
            for i in range(MAX_WARMUP_STEPS):
                best_child = None
                best_affinity = -np.inf
                for _ in range(10):
                    mutated_seq, num_mutations = self._mutate_sequence(current_seq, warmup_regions_mu)
                    mutated_a_seq, min_stop, frame = translate_nucleotide_to_protein_min_stops_cached(mutated_seq)
                    if min_stop > 0:
                        continue
                    temp = {"sequence": mutated_seq, "a_sequence": mutated_a_seq}
                    affinity = self._pssm_score(pssm, pssm_mode, temp)
                    if affinity > best_affinity:
                        best_affinity  = affinity
                        best_child     = (mutated_seq, mutated_a_seq, frame, num_mutations)

                if best_child is None:
                    continue

                current_seq, current_a_seq, current_frame, step_mutations = best_child
                total_warmup_mutations += step_mutations
                if step_mutations > 0:
                    if i==0:
                        prnt_id = 'germline'
                    else:
                        prnt_id = f"wu_seq_{i}"
                    temp_seq = {
                        "id":         f"wu_seq_{i+1}",
                        "sequence":   current_seq,
                        "a_sequence": current_a_seq,
                        "frame":      current_frame,
                        "generation": None,
                        "parent":     prnt_id,
                        "mutations":  step_mutations,
                        "affinity":   best_affinity,
                        "abundance":  None,
                    }
                    warmup_path.append(temp_seq)
                if best_affinity >= warmup_thresh:
                    logger.info(
                        f"Warmup converged after {i+1} steps | "
                        f"total mutations={total_warmup_mutations} | "
                        f"affinity={best_affinity:.4f}"
                    )
                    break
            else:
                logger.warning(
                    f"Warmup did not reach threshold after {MAX_WARMUP_STEPS} steps. "
                    f"Best affinity={best_affinity:.4f} | threshold={warmup_thresh:.4f}. Skipping run."
                )
                return [], []

            # Build seq0 as a child of root, with accumulated mutations
            seq0 = {
                "id":         "seq0",
                "sequence":   current_seq,
                "a_sequence": current_a_seq,
                "frame":      current_frame,
                "generation": 1,
                "parent":     "naive",
                "mutations":  total_warmup_mutations,
                "affinity":   best_affinity,
                "abundance":  0,
            }
            all_bcrs          = [root, seq0]
            current_generation = [seq0]
        else:
            all_bcrs           = [root]
            current_generation = [root]

        # ---- Main simulation (unchanged) ----
        generation = 2 if root["affinity"] < thresh else 1
        while generation <= max_generations:
            next_generation = []

            for parent in current_generation:
                for _ in range(2):
                    mutated_seq, num_mutations = self._mutate_sequence(parent["sequence"], pssm.regions_mu)

                    if num_mutations > 0:
                        mutated_a_seq, min_stop, frame = \
                            translate_nucleotide_to_protein_min_stops_cached(mutated_seq)
                    else:
                        mutated_a_seq = parent["a_sequence"]
                        min_stop      = mutated_a_seq.count('*')
                        frame         = parent["frame"]

                    temp_child = {
                        "sequence":   mutated_seq,
                        "a_sequence": mutated_a_seq,
                    }
                    affinity = self._pssm_score(pssm, pssm_mode, temp_child)

                    if affinity >= thresh and min_stop == 0:
                        child = {
                            "id":         self._generate_bcr_id(),
                            "sequence":   mutated_seq,
                            "a_sequence": mutated_a_seq,
                            "frame":      frame,
                            "generation": generation,
                            "parent":     parent["id"],
                            "affinity":   affinity,
                            "abundance":  0,
                            "mutations":  num_mutations,
                        }
                        all_bcrs.append(child)
                        next_generation.append(child)

            if not next_generation:
                break

            current_generation = next_generation
            generation        += 1
            if len({d['sequence'] for d in all_bcrs}) >= minimum_num_unique_sequences:
                break

        parents_set = {item['parent'] for item in all_bcrs if item.get('parent') is not None}
        for item in all_bcrs:
            if item.get('id') not in parents_set:
                item['abundance'] = 1
        all_bcrs        = clean_sequences(all_bcrs)
        merged_all_bcrs = merge_sequences(all_bcrs)
        return all_bcrs, merged_all_bcrs, warmup_path




    # def generate_repertoire(self, pssm, pssm_mode, max_generations, minimum_num_unique_sequences, naive_nuc_sequence):

    #     root = self.build_root_bcr(naive_nuc_sequence)
    #     root["affinity"] = self._pssm_score(pssm, pssm_mode, root)
    #     all_bcrs = [root]
    #     current_generation = [root]
    #     generation = 1
    #     thresh = np.min(pssm.kde.dataset)
    #     while generation <= max_generations:
    #         next_generation = []

    #         for parent in current_generation:
    #             for _ in range(2):                      # each cell divides into 2
    #                 mutated_seq, num_mutations = self._mutate_sequence(parent["sequence"], pssm.regions_mu)

    #                 if num_mutations > 0:
    #                     mutated_a_seq, min_stop, frame = \
    #                         translate_nucleotide_to_protein_min_stops_cached(mutated_seq)
    #                 else:
    #                     mutated_a_seq = parent["a_sequence"]
    #                     min_stop      = mutated_a_seq.count('*')
    #                     frame         = parent["frame"]

    #                 # Build a temporary child dict for scoring
    #                 temp_child = {
    #                     "sequence":   mutated_seq,
    #                     "a_sequence": mutated_a_seq,
    #                 }
    #                 affinity = self._pssm_score(pssm, pssm_mode, temp_child)

    #                 #thresh = min(1.0, pssm.kde.resample(1).item())

    #                 if affinity >= thresh and min_stop == 0:
    #                     child = {
    #                         "id":         self._generate_bcr_id(),
    #                         "sequence":   mutated_seq,
    #                         "a_sequence": mutated_a_seq,
    #                         "frame":      frame,
    #                         "generation": generation,
    #                         "parent":     parent["id"],
    #                         "affinity":   affinity,
    #                         "abundance":  0,
    #                         "mutations":  num_mutations,
    #                     }
    #                     all_bcrs.append(child)
    #                     next_generation.append(child)

    #         if not next_generation:
    #             break

    #         current_generation = next_generation
    #         generation        += 1
    #         if len({d['sequence'] for d in all_bcrs}) >= minimum_num_unique_sequences:
    #             break
    #     parents_set = {item['parent'] for item in all_bcrs if item.get('parent') is not None}
    #     for item in all_bcrs:
    #         if item.get('id') not in parents_set:
    #             item['abundance'] = 1
    #     all_bcrs        = clean_sequences(all_bcrs)
    #     merged_all_bcrs = merge_sequences(all_bcrs)
    #     return all_bcrs, merged_all_bcrs

    # ------------------------------------------------------------------
    # PSSM scoring dispatcher
    # ------------------------------------------------------------------

    @staticmethod
    def _pssm_score(pssm, pssm_mode, bcr_dict):
        """Return PSSM score for *bcr_dict* according to *pssm_mode*."""
        if pssm_mode == 'nucleotide':
            return pssm.score(bcr_dict["sequence"])
        else:   # 'aminoacid'
            return pssm.score(bcr_dict["a_sequence"])


# ---------------------------------------------------------------------------
# Utility functions  (unchanged from original unless noted)
# ---------------------------------------------------------------------------

def translate_nucleotide_to_protein_min_stops(nuc_seq):
    seq_obj    = Seq(nuc_seq.upper().replace('U', 'T'))
    best_protein = None
    min_stops    = float('inf')
    best_frame   = 0
    for frame in range(3):
        protein_seq  = seq_obj[frame:].translate(to_stop=False)
        protein_str  = str(protein_seq)
        total_stops  = protein_seq.count('*')
        if total_stops < min_stops:
            min_stops    = total_stops
            best_protein = protein_str
            best_frame   = frame
    return best_protein, min_stops, best_frame


def biased_mutation(base, p_trans):
    transitions  = {'a': 'g', 'g': 'a', 'c': 't', 't': 'c'}
    transversions = {
        'a': ['c', 't'],
        'g': ['c', 't'],
        'c': ['a', 'g'],
        't': ['a', 'g'],
    }
    base = base.lower()
    if random.random() < p_trans:
        return transitions.get(base, base)
    else:
        return random.choice(transversions.get(base, [base]))

def load_sequences(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    seqs = [str(rec.seq).upper() for rec in records]
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("All sequences must have the same length")

    abundances = []
    for rec in records:
        name = rec.id
        abund = int(name.split("@")[1]) if "@" in name else 1
        abundances.append(abund)

    char_map = np.zeros(256, dtype=np.uint8)
    for i, c in enumerate(b'ACGTN'):
        char_map[c] = i + 1
    arr = np.array([[char_map[ord(c)] for c in s] for s in seqs], dtype=np.uint8)

    return arr, seqs, abundances

def compute_hamming_matrix(arr, chunk=500):
    n, L = arr.shape
    if n <= 1000:
        diff = arr[:, None, :] != arr[None, :, :]
        return np.sum(diff, axis=2).astype(np.uint16)
    W = np.zeros((n, n), dtype=np.uint16)
    for i in range(0, n, chunk):
        sl = arr[i:i+chunk]
        diff = sl[:, None, :] != arr[None, :, :]
        W[i:i+chunk, :] = diff.sum(axis=2)
    return W

def calc_mut_rate(mst_edges, arr, abundances):
    """
    arr here can be the full array or a region slice — same logic either way.
    """
    idx_map = np.full(6, -1, dtype=np.int8)
    idx_map[1] = 0; idx_map[2] = 1; idx_map[3] = 2; idx_map[4] = 3

    ti_matrix = np.zeros((4, 4), dtype=bool)
    ti_matrix[0, 2] = ti_matrix[2, 0] = True  # A<->G
    ti_matrix[1, 3] = ti_matrix[3, 1] = True  # C<->T

    total_ti = 0
    total_tv = 0
    for u, v in mst_edges:
        u_idx = idx_map[arr[u]]
        v_idx = idx_map[arr[v]]
        valid_diff = (u_idx != v_idx) & (u_idx >= 0) & (v_idx >= 0)
        u_diff = u_idx[valid_diff]
        v_diff = v_idx[valid_diff]
        ti_count = np.sum(ti_matrix[u_diff, v_diff])
        total_ti += ti_count
        total_tv += len(u_diff) - ti_count

    total_muts = total_ti + total_tv
    if total_muts == 0:
        return 0, 0, 0
    p_rate = total_muts / ((sum(abundances) - 1) * arr.shape[1])  # arr.shape[1] = region length
    p_ti = total_ti / total_muts
    p_tv = total_tv / total_muts
    return p_rate, p_ti, p_tv

def calc_regions_mut_rate(arr, abundances, regions):
    
    W = compute_hamming_matrix(arr)

    G = nx.from_numpy_array(W)
    mst = nx.minimum_spanning_tree(G, weight='weight')
    mst_edges = list(mst.edges())

    # Compute mutation stats per region by slicing arr columns
    results = {}
    for region, (start, end) in regions.items():
        arr_region = arr[:, start:end]  # shape (n, region_length)
        p_rate, p_ti, p_tv = calc_mut_rate(mst_edges, arr_region, abundances)
        results[region] = {"start": start, "end": end, "p_rate": p_rate, "p_ti": p_ti, "p_tv": p_tv}
    
    L = arr.shape[1]

    intervals = sorted(regions.values())  # sort by start

    keep = []
    prev = 0
    for start, end in intervals:
        if prev < start:
            keep.extend(range(prev, start))
        prev = end
    if prev < L:
        keep.extend(range(prev, L))

    if keep:
        arr_none = arr[:, keep]
        p_rate, p_ti, p_tv = calc_mut_rate(mst_edges, arr_none, abundances)

        results["none_region"] = {
            "positions": keep,
            "p_rate": p_rate,
            "p_ti": p_ti,
            "p_tv": p_tv
        }

    return results

def clean_sequences(seq_list):
    id_map     = {seq.get("id"): seq for seq in seq_list if "id" in seq}
    to_remove  = []

    for seq in seq_list:
        if seq['id'] == 'naive':
            continue
        if seq.get("mutations", None) == 0:
            parent_id = seq.get("parent")
            abundance = seq.get("abundance", 0)
            if parent_id in id_map:
                id_map[parent_id]["abundance"] = \
                    id_map[parent_id].get("abundance", 0) + abundance
            for child in seq_list:
                if child.get("parent") == seq.get("id"):
                    child["parent"] = parent_id
            to_remove.append(seq)
    remove_ids = {seq['id'] for seq in to_remove}
    return [s for s in seq_list if s['id'] not in remove_ids]


def merge_sequences(seq_list):
    grouped = defaultdict(list)
    for entry in seq_list:
        key = (entry['sequence'], entry['parent'])
        grouped[key].append(entry)

    merged_map  = {}
    merged_list = []

    for entries in grouped.values():
        if len(entries) == 1:
            merged_list.append(entries[0])
        else:
            entries.sort(key=lambda x: int(x['id'][3:]) if x['id'] != 'naive' else -1)
            merged_id      = entries[0]['id']
            total_abundance = sum(e['abundance'] for e in entries)
            merged_entry   = entries[0].copy()
            merged_entry['id']        = merged_id
            merged_entry['abundance'] = total_abundance
            merged_list.append(merged_entry)
            for e in entries:
                merged_map[e['id']] = merged_id

    for entry in merged_list:
        parent = entry['parent']
        if parent in merged_map:
            entry['parent'] = merged_map[parent]

    for entry in merged_list:
        if entry['id'] not in merged_map:
            merged_map[entry['id']] = entry['id']

    for entry in merged_list:
        if entry['parent'] in merged_map:
            entry['parent'] = merged_map[entry['parent']]

    return merged_list


def calculate_stats(bcrs):
    if not bcrs:
        return None
    affinities  = np.array([b['affinity']   for b in bcrs])
    generations = np.array([b['generation'] for b in bcrs])
    abundances  = np.array([b['abundance']  for b in bcrs])
    unique_seqs = set(b['sequence'] for b in bcrs)
    return {
        "naive_affinity":       bcrs[0]['affinity'],
        "mean_affinity":        float(np.mean(affinities)),
        "median_affinity":      float(np.median(affinities)),
        "max_affinity":         float(np.max(affinities)),
        "min_affinity":         float(np.min(affinities)),
        "mean_abundance":       float(np.mean(abundances)),
        "max_abundance":        int(np.max(abundances)),
        "min_abundance":        int(np.min(abundances)),
        "total_abundance":      int(np.sum(abundances)),
        "max_generation":       int(np.max(generations)),
        "num_bcrs":             len(bcrs),
        "num_unique_sequences": len(unique_seqs),
    }


# ---------------------------------------------------------------------------
# Export helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def export_to_fasta(bcrs, filename):
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_available{ext}"
    with open(filename, 'w') as f, open(new_filename, 'w') as f_new:
        for b in bcrs:
            f.write(f">{b['id']}@{b['abundance']}\n{b['sequence']}\n")
            if b['id']=='naive' or b.get('abundance', 0) > 0:
                f_new.write(f">{b['id']}@{b['abundance']}\n{b['sequence']}\n")


def get_cleaned_bcrs_for_newick(bcrs):
    new_data = copy.deepcopy(bcrs)
    id_to_entry = {d['id']: d for d in new_data}
    zero_ids = [d['id'] for d in new_data if (d['abundance'] == 0 and d['id']!='naive')]

    for zid in zero_ids:
        zero_entry = id_to_entry.get(zid)
        if zero_entry is None:
            continue
        grandparent = zero_entry['parent']   # may be None
        for d in new_data:
            if d['parent'] == zid:
                d['parent'] = grandparent

    return [d for d in new_data if d['abundance'] != 0 or d['id'] == 'naive']


def export_to_newick(bcrs, filename):
    tree       = defaultdict(list)
    seq_map    = {}
    parent_map = {}

    for b in bcrs:
        b_id   = b['id']
        parent = b['parent']
        if parent:
            tree[parent].append(b_id)
        seq_map[b_id]    = b['sequence']
        parent_map[b_id] = parent

    roots = [b['id'] for b in bcrs if b['parent'] is None]
    if not roots:
        return None

    def mutation_distance(s1, s2):
        return sum(a != b for a, b in zip(s1, s2))

    def recurse(node_id):
        children = tree[node_id]
        label    = node_id if parent_map[node_id] is not None else "naive"
        dist     = (0 if parent_map[node_id] is None
                    else mutation_distance(seq_map[parent_map[node_id]], seq_map[node_id]))
        if not children:
            return f"{label}:{dist}"
        subtree = ",".join(recurse(c) for c in children)
        return f"({subtree}){label}:{dist}"

    newick = '(' + recurse(roots[0]) + ');'
    with open(filename, 'w') as f:
        f.write(newick)
    return newick


# ---------------------------------------------------------------------------
# Tree visualisation  (unchanged from original)
# ---------------------------------------------------------------------------

def plot_newick_bcellTree(newick, repertoire, output_file,
                          title_name='Run', title_num=0):
    node_weights = {item['id']: item['abundance'] for item in repertoire}
    def rectangular_layout(tree):
        positions     = {}
        x_offset      = 0
        level_spacing  = 50
        sibling_spacing = 100

        def assign_positions(node, x, y):
            nonlocal x_offset
            if node.is_leaf():
                positions[node.name] = (x_offset, y)
                x_offset += sibling_spacing
            else:
                child_positions = []
                for child in node.children:
                    assign_positions(child, x, y - level_spacing)
                    child_positions.append(positions[child.name][0])
                positions[node.name] = (
                    sum(child_positions) / len(child_positions), y
                )

        assign_positions(tree, x_offset, 0)
        return positions

    node_info = {
        key: f'Node_name: {key}\n\nAbundancy: {int(val)}'
        for key, val in node_weights.items()
    }
    t          = Tree(newick, format=1)
    tree_nodes = {node.name for node in t.traverse() if node.name}
    node_info.update({
        key: 'Node_name: \n\nAbundancy: 1'
        for key in tree_nodes if key not in node_info
    })
    node_weights.update({key: 1 for key in tree_nodes if key not in node_weights})

    max_weight   = max(node_weights.values())
    node_sizes   = {n: (w / max_weight) * 1000 for n, w in node_weights.items()}
    weights      = np.array(list(node_weights.values()))
    norm_weights = (
        np.ones_like(weights) if weights.max() == weights.min()
        else (weights - weights.min()) / (weights.max() - weights.min())
    )
    colors      = cm.viridis(norm_weights)
    node_colors = {n: colors[i] for i, n in enumerate(node_weights.keys())}
    pos         = rectangular_layout(t)

    fig = plt.figure(figsize=(8, 5))
    fig.canvas.manager.set_window_title(f'{title_name} T{title_num + 1}')

    x_coords = [x for x, y in pos.values()]
    y_coords = [y for x, y in pos.values()]
    x_range  = max(x_coords) - min(x_coords)
    y_range  = max(y_coords) - min(y_coords)
    x_margin = x_range * 0.3 if x_range > 0 else 1.0
    y_margin = y_range * 0.3 if y_range > 0 else 1.0
    plt.gca().set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
    plt.gca().set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)

    scatter = plt.scatter([], [], s=[], alpha=0.9, color=[], edgecolor="black", zorder=2)
    scatter.set_offsets([pos[n] for n in tree_nodes])
    scatter.set_sizes([node_sizes[n] for n in tree_nodes])
    scatter.set_color([node_colors[n] for n in tree_nodes])

    for node in tree_nodes:
        if node == "naive":
            plt.scatter(pos[node][0], pos[node][1],
                        s=20, alpha=0.9, color="black",
                        edgecolor="black", zorder=2, marker="^")

    for node in t.traverse("postorder"):
        if not node.is_root():
            parent       = node.up
            x_s, y_s    = pos[parent.name]
            x_e, y_e    = pos[node.name]
            plt.plot([x_s, x_e], [y_s, y_s], color="black", lw=1, zorder=1)
            plt.plot([x_e, x_e], [y_s, y_e], color="black", lw=1, zorder=1)

    plt.axis("off")
    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        node_name = list(tree_nodes)[sel.index]
        info_text = node_info.get(node_name, "No information available")
        sel.annotation.set_text(info_text)
        sel.annotation.set_multialignment('left')
        sel.annotation.set_bbox(dict(
            boxstyle="round,pad=0.7", edgecolor="black",
            facecolor="yellow", linewidth=1, alpha=0.7
        ))
        sel.annotation.arrowprops = None

    plt.savefig(output_file, format='png')
    plt.close()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_run_metadata(run_dir, args, stats, stats_available,success):
    metadata = {
        "timestamp":  datetime.now().isoformat(),
        "parameters": vars(args),
        "stats":      stats,
        "available_repertoire_stat": stats_available,
        "success":    success,
    }
    with open(os.path.join(run_dir, "run_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# PSSM construction from input file
# ---------------------------------------------------------------------------

def build_pssm_from_file(sequences_path, regions, pssm_mode, pssm_scoring, background):
    """
    Read aligned nucleotide sequences from *sequences_path*, optionally
    translate them, and return a PSSM object.
    """
    #raw_seqs = read_fasta(sequences_path)
    arr, raw_seqs, abundances = load_sequences(sequences_path)
    if not raw_seqs:
        logger.error(f"No sequences found in {sequences_path}")
        sys.exit(1)

    # Normalise to lowercase nucleotide
    nuc_seqs = [s.lower().replace('u', 't') for s in raw_seqs]

    if pssm_mode == 'nucleotide':
        alphabet   = NUC_ALPHABET
        input_seqs = nuc_seqs
        logger.info(f"Building nucleotide PSSM from {len(input_seqs)} sequences")
    else:   # 'aminoacid'
        input_seqs = []
        for ns in nuc_seqs:
            aa, _, _ = translate_nucleotide_to_protein_min_stops(ns)
            input_seqs.append(aa.upper())
        alphabet = AA_ALPHABET
        logger.info(f"Building amino-acid PSSM from {len(input_seqs)} sequences")

    # All sequences must be the same length for a proper PSSM
    lengths = set(len(s) for s in input_seqs)
    if len(lengths) > 1:
        logger.warning(
            f"Input sequences have varying lengths {lengths}. "
            "PSSM width will be set to the minimum length; "
            "longer sequences will be windowed at score time."
        )

    regions_mut_rates = calc_regions_mut_rate(arr, abundances, regions)
    return PSSM(input_seqs, alphabet, regions_mut_rates, scoring=pssm_scoring, background=background)

def extract_germline(fasta_path, airr_path):
    ids = set()
    with Path(fasta_path).open() as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])


    df = pd.read_csv(Path(airr_path), sep="\t")
    matched = df[df["sequence_id"].isin(ids)].copy()
    if matched.empty:
        raise RuntimeError("No AIRR rows matched the FASTA sequence IDs.")


    def clean(s):
        return (s or "").replace(".", "").replace("-", "").upper()

    def germline_seq(row):
        return clean(str(row.get("v_germline_alignment") or "")
                   + str(row.get("np1") or "")
                   + str(row.get("d_germline_alignment") or "")
                   + str(row.get("np2") or "")
                   + str(row.get("j_germline_alignment") or ""))

    def to_float(val):
        try:
            return float(val or 0)
        except (ValueError, TypeError):
            return 0.0

    matched["_germline_seq"] = matched.apply(germline_seq, axis=1)
    matched["_abundance"] = matched["sequence_id"].apply(
        lambda sid: int(sid.rsplit("@", 1)[1]) if "@" in sid else 1
    )

    candidates = matched[matched["_germline_seq"] != ""]
    if candidates.empty:
        raise RuntimeError("No reconstructable germline sequence found.")


    best = candidates.loc[
        candidates.apply(
            lambda r: (to_float(r.get("v_identity")),
                       to_float(r.get("j_identity")),
                       to_float(r.get("d_identity")),
                       r["_abundance"]),
            axis=1,
        ).idxmax()
    ]


    try:
        offset = int(float(best.get("v_sequence_start") or 1))
    except (ValueError, TypeError):
        offset = 1

    positions = {}
    for field in REGION_FIELDS:
        val = best.get(field, "")
        try:
            positions[field] = int(float(val)) - offset
        except (ValueError, TypeError):
            positions[field] = ""

    return best["_germline_seq"], positions
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ancestra-Echo",
        epilog="Example: python pssm_bcr_simulator.py --naive naive.fasta "
               "--sequences set_X.fasta --clones 5 --max-gen 20 --min-seq 50"
    )

    parser.add_argument(
        "--sequences", type=str, default="sequences.fasta",
        help="FASTA file containing the aligned nucleotide sequences"
             "(used to build the PSSM)"
    )

    parser.add_argument(
        "--airr", type=str, default="vquest_airr.tsv",
        help="airr data in .tsv"
    )

    # ---- PSSM options ----
    parser.add_argument(
        "--pssm-mode", type=str, default="nucleotide",
        choices=["nucleotide", "aminoacid"],
        help="Build and score the PSSM at nucleotide or amino-acid level (default: nucleotide)"
    )
    parser.add_argument(
        "--pssm-scoring", type=str, default="logodds",
        choices=["logodds", "frequency"],
        help="Scoring scheme for the PSSM (default: logodds)"
    )
    parser.add_argument(
        "--background", type=str, default="observed",
        choices=["uniform", "observed"],
        help="Background frequency for log-odds scoring (default: observed)"
    )

    # ---- Simulation parameters ----
    parser.add_argument(
        "--clones", type=int, default=1,
        help="Number of independent successful clones to generate (default: 1)"
    )
    parser.add_argument(
        "--max-gen", type=int, default=40,
        help="Maximum number of generations per simulation run (default: 15)"
    )
    parser.add_argument(
        "--min-seq", type=int, default=100,
        help="Minimum unique sequences required to accept a run (default: 10)"
    )

    # ---- Output ----
    parser.add_argument(
        "--output-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
        help="Base directory for simulation outputs"
    )
    parser.add_argument(
        "--plot-tree", action="store_true",
        help="Generate lineage tree PNG for successful runs"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ---- Load naive sequence ----

    naive_records, positions = extract_germline(args.sequences, args.airr)

    naive_nuc_sequence = naive_records.lower().replace('u', 't')
    regions = {
        "FR1":  (positions["fwr1_start"], positions["fwr1_end"] + 1),
        "CDR1": (positions["cdr1_start"], positions["cdr1_end"] + 1),
        "FR2":  (positions["fwr2_start"], positions["fwr2_end"] + 1),
        "CDR2": (positions["cdr2_start"], positions["cdr2_end"] + 1),
        "FR3":  (positions["fwr3_start"], positions["fwr3_end"] + 1),
        "CDR3": (positions["cdr3_start"], positions["cdr3_end"] + 1),
        "FR4":  (positions["fwr4_start"], positions["fwr4_end"] + 1),
    }
    # ---- Build PSSM ----
    pssm = build_pssm_from_file(
        args.sequences,
        regions = regions,
        pssm_mode = args.pssm_mode,
        pssm_scoring = args.pssm_scoring,
        background = args.background,
    )

    # ---- Simulator ----
    sim = BCRSimulator()

    successful_runs = 0
    attempts        = 0
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    while successful_runs < args.clones:
        attempts += 1
        repertoire, merged_repertoire, warmup_path = sim.generate_repertoire(
            pssm = pssm,
            pssm_mode = args.pssm_mode,
            max_generations = args.max_gen,
            minimum_num_unique_sequences = args.min_seq,
            naive_nuc_sequence = naive_nuc_sequence,
        )
        available_merged_repertoire = get_cleaned_bcrs_for_newick(merged_repertoire)
        stats   = calculate_stats(merged_repertoire)
        stats_available = calculate_stats(available_merged_repertoire)
        if stats_available is None:
            success = False
        else:
            success = (stats_available['num_unique_sequences'] >= args.min_seq)

        if success:
            successful_runs += 1
            run_folder = (base_output_dir / "successful_clones" /
                            f"run_{successful_runs}_gen{args.max_gen}_minSeq{args.min_seq}")
        else:
            run_folder = (base_output_dir / "rejected_clones" /
                            f"rejected_attempt_{attempts}")

        run_folder.mkdir(parents=True, exist_ok=True)

        export_to_fasta(merged_repertoire, run_folder / "repertoire.fasta")
        newick = export_to_newick(merged_repertoire, run_folder / "repertoire.nk")
        newick_available = export_to_newick(available_merged_repertoire, run_folder / "repertoire_available.nk")


        if args.plot_tree and success:
            plot_newick_bcellTree(
                newick, merged_repertoire,
                run_folder / "lineage_tree.png",
                title_name=f"RUN {successful_runs}",
                title_num=successful_runs,
            )
            plot_newick_bcellTree(
                newick_available, available_merged_repertoire,
                run_folder / "lineage_tree_available.png",
                title_name=f"RUN {successful_runs}",
                title_num=successful_runs,
            )

        # Save per-sequence CSV
        keys_to_include = ['id', 'a_sequence', 'frame', 'generation',
                            'parent', 'mutations', 'affinity', 'abundance']
        with open(run_folder / "repertoire_info.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys_to_include)
            writer.writeheader()
            for entry in merged_repertoire:
                writer.writerow({k: entry[k] for k in keys_to_include if k in entry})

        with open(run_folder / "available_repertoire_info.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys_to_include)
            writer.writeheader()
            for entry in available_merged_repertoire:
                writer.writerow({k: entry[k] for k in keys_to_include if k in entry})

        if warmup_path:
            keys_to_include = ['id', 'sequence', 'a_sequence', 'parent', 'mutations', 'affinity']
            with open(run_folder / "warmup_info.csv", 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys_to_include)
                writer.writeheader()
                for entry in warmup_path:
                    writer.writerow({k: entry[k] for k in keys_to_include if k in entry})

        if stats is None or stats_available is None:
            save_run_metadata(run_folder, args, None, None, success)
        else:
            save_run_metadata(run_folder, args, stats, stats_available, success)


if __name__ == "__main__":
    main()
