# User Manual: Running Ancestra-Echo

## Overview
Ancestra-Echo is a command-line Python program for generating simulated B-cell receptor (BCR) repertoires from input sequence data.
The program requires:
*	a FASTA file containing aligned nucleotide sequences
*	an AIRR-format annotation file (.tsv) containing germline information
The program produces simulated repertoires, lineage trees, sequence information tables, and run metadata.

## Requirements
Python 3.9+ is recommended.

Install dependencies:

```bash
pip install numpy pandas scipy biopython matplotlib ete3 networkx mplcursors
```

## Input Files

### FASTA file (`--sequences`)
This file contains aligned nucleotide sequences used as the input repertoire.

Example:
```text
>seq1@10
ATGGCCTAGCTAGCTAGC
>seq2@5
ATGGCCTAGCTAGATAGC
>seq3@2
ATGGCCTAGCTAGCTTGC

```
Requirements:
*	File format: FASTA
*	Sequences must be aligned.
*	All sequences should have the same length.
*	Sequence identifiers may include abundance information using:
sequence_id@abundance
Example:
```text
>seqence1@25
```
means abundance = 25.
If no abundance is provided, abundance is assumed to be 1.
Default filename:
```text
sequences.fasta
```

### AIRR file (`--airr`)
This file contains AIRR-format annotation information.
Format:
*	Tab-separated values (.tsv)
*	Must include sequence identifiers matching the FASTA headers.
Required information includes germline alignment fields and region annotations.
Default filename:
```text
vquest_airr.tsv
```

## Running
The program is executed from the command line:
```bash
python ancestra_echo_v1.py [options]
```

Example:

```bash
python ancestra_echo_v1.py --sequences dominant_clones.fasta --airr vquest_airr.tsv --clones 5 --max-gen 20 --min-seq 50
```

## Parameters

### Input
- `--sequences`: FASTA input file.
- `--airr`: AIRR TSV annotation file.

### PSSM options
Defines whether scoring is performed using nucleotide or amino-acid sequences.
- `--pssm-mode`: `nucleotide` or `aminoacid`.
- `--pssm-scoring`: `logodds` or `frequency`.
- `--background`: `uniform` or `observed`.

### Simulation options
- `--clones`: Number of successful simulations.
- `--max-gen`: Maximum generations.
- `--min-seq`: Minimum unique sequences.

### Output options
- `--output-dir`: Output directory.
- `--plot-tree`: Generate lineage tree PNG files.
- `--verbose`: Enable detailed logging.

## Outputs

Generated files include:

- `repertoire.fasta`: all (extinct and existing) simulated BCR nucleotide sequences.
- `repertoire_available.fasta`: all available (existing) simulated BCR nucleotide sequences.
- `repertoire_info.csv`: all sequence information.
- `available_repertoire_info.csv`: available sequence information.
- `repertoire.nk`: Newick lineage tree.
- `repertoire_available.nk`: Newick lineage tree related to available sequences.
- `lineage_tree.png`: tree visualization (with `--plot-tree`).
- `run_metadata.json`: run parameters and statistics.

After execution, results are stored in:
output_directory/
│
├── successful_clones/
│   └── run_1_gen40_minSeq100/
│       ├── repertoire.fasta
│       ├── repertoire_available.fasta
│       ├── repertoire.nk
│       ├── repertoire_available.nk
│       ├── repertoire_info.csv
│       ├── available_repertoire_info.csv
│       ├── run_metadata.json
│       ├── lineage_tree_available.png
│       └── lineage_tree.png
│
└── rejected_clones/
    └── rejected_attempt_1/
        ├── repertoire.fasta
        ├── repertoire.nk
        ├── repertoire_info.csv
        └── run_metadata.json


## Example Complete Command

```bash
python ancestra_echo_v1.py \
--sequences dominant_clones.fasta \
--airr vquest_airr.tsv \
--pssm-mode nucleotide \
--clones 5 \
--max-gen 40 \
--min-seq 100 \
--output-dir simulation_results \
--plot-tree
```


### Sequences have different lengths
Ensure FASTA sequences are aligned before running.

### AIRR IDs do not match
Ensure FASTA headers match AIRR `sequence_id` values.

### No successful clones
Try increasing `--max-gen` or reducing `--min-seq`.

## Help
To display available command-line options:

```bash
python pssm_bcr_simulator.py --help
```
## Contact

For questions, bug reports, or contributions, please contact the repository maintainer, **Reza Hassanzadeh**, at **rhz.sbu@gmail.com**.
