import logging
from pathlib import Path
from itertools import repeat
from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from seqwin import config, load
from seqwin.markers import ConnectedKmers
from seqwin.ncbi import blast
from seqwin.utils import StartMethod, mkdir, run_cmd, mp_wrapper

from msa_tools import MSA

logger = logging.getLogger(__name__)

BLAST_DB = Path('core_nt')
BLAST_COL = ( # Default columns to be included in the BLAST tsv output
    'qseqid', # 1. query or source (e.g., gene) sequence id
    'saccver', # 2. subject or target (e.g., reference genome) sequence id
    'length', # 3. alignment length
    'nident', # 4. number of identical matches (= length - mismatch - gaps)
    'mismatch', # 5. number of mismatches
    'gaps', # 6. total number of gaps
    'bitscore', # 7. bit score
    'sseq', # 8. aligned part of subject sequence
)
MSA_DIR = 'msa'
SUFFIX_TAR = 'tar'
SUFFIX_NEG = 'neg'
VARVAMP_DIR = 'varvamp'
VARVAMP_DESIGN = 'qpcr_design.tsv'
VARVAMP_PRIMERS = 'qpcr_primers.tsv'
METRICS = {
    'log10_hits': -1, 
    'varVAMP_p': -1, 
    'sensitivity': 1, 
    'specificity': 1, 
}


def mafft(seq_list: Iterable[str], header_list: Iterable[str] | None=None, n_cpu: int=1) -> str:
    if header_list is None:
        header_list = range(len(seq_list))

    fasta = ''.join(
        list(f'>{h}\n{s}\n' for h, s in zip(header_list, seq_list))
    )
    return run_cmd(
        'mafft', '--auto', '--quiet', '--thread', str(n_cpu), '-', 
        stdin=fasta
    ).stdout


def _msa_worker(i: int, blast_out: pd.DataFrame, msa_prefix: Path) -> Path:
    tar = blast_out[blast_out['is_target']]
    tar_msa_path = msa_prefix / Path(f'{i}-{SUFFIX_TAR}.fasta')
    tar_msa_path.write_text(
        mafft(tar['sseq'], tar['assembly_idx'], 1)
    )

    neg = blast_out[~blast_out['is_target']]
    neg_msa_path = msa_prefix / Path(f'{i}-{SUFFIX_NEG}.fasta')
    neg_msa_path.write_text(
        mafft(neg['sseq'], neg['assembly_idx'], 1)
    )

    return tar_msa_path


def varvamp(t: float, a: int, msa_path: Path, out_path: Path) -> None:
    run_cmd(
        'varvamp', 'qpcr', '-t', str(t), '-a', str(a), '-th', '1', msa_path, out_path, 
        raise_error=False
    )


def _blast_nt(markers: list[ConnectedKmers], taxid: int, prefix: Path, n_cpu: int) -> NDArray:
    logger.info(f'BLAST checking signatures against {BLAST_DB}...')
    blast_out = blast(
        list(marker.rep.seq for marker in markers), 
        db=BLAST_DB, columns=BLAST_COL, neg_taxids=[taxid], n_cpu=n_cpu
    )

    # save BLAST outputs
    blast_out.to_pickle(prefix / f'blast.pkl')

    # count BLAST hits
    n_hits = np.zeros(len(markers), dtype=np.intp)
    # some signatures might have no hit
    for i, c in blast_out['qseqid'].value_counts(sort=False).items():
        n_hits[i] = c

    return n_hits


def _eval_designs(
    varvamp_prefix: Path, msa_prefix: Path, n_markers: int, n_tar: int, n_neg: int, n_cpu: int
) -> NDArray:
    logger.info('Evaluating each varVAMP assay...')
    n_cpu = min(n_cpu, 8)

    metrics = np.full((n_markers, 3), np.nan) # varVAMP penalty; sensitivity; specificity
    for i in range(n_markers):
        curr_design = varvamp_prefix / Path(str(i))
        try:
            df = pd.read_csv(curr_design / VARVAMP_DESIGN, sep='\t')
        except FileNotFoundError:
            # no design for this signature
            continue

        # get the name of the best design (lowest penalty)
        df.sort_values('penalty', inplace=True, ignore_index=True)
        metrics[i, 0] = df['penalty'][0]
        design = df['qpcr_scheme'][0]

        # get the oligo sequences of the best design
        df = pd.read_csv(curr_design / VARVAMP_PRIMERS, sep='\t')
        design = df[df['qpcr_scheme'] == design]
        oligos = design['seq']

        # sensitivity
        tar_msa_path = msa_prefix / Path(f'{i}-{SUFFIX_TAR}.fasta')
        tar_msa = MSA(tar_msa_path, n_cpu)
        tar_match = list(
            tar_msa.attach_oligo(o) for o in oligos
        )
        metrics[i, 1] = sum(np.logical_and.reduce(tar_match)) / n_tar

        # specificity
        neg_msa_path = msa_prefix / Path(f'{i}-{SUFFIX_NEG}.fasta')
        try:
            neg_msa = MSA(neg_msa_path, n_cpu)
            neg_match = list(
                neg_msa.attach_oligo(o) for o in oligos
            )
            # use logical_or for neg
            metrics[i, 2] = 1 - sum(np.logical_or.reduce(neg_match)) / n_neg
        except IndexError:
            # MSA file is empty
            metrics[i, 2] = 1.

    return metrics


def _standardize(s: pd.Series) -> pd.Series:
    return (s - s.mean(skipna=True)) / s.std(skipna=True, ddof=1)


def _get_scores(designs: pd.DataFrame) -> pd.Series:
    scores = pd.Series(.0, index=range(len(designs)))
    for k, w in METRICS.items():
        scores += w * _standardize(designs[k])
    return scores


def main(seqwin_out: Path, taxid: int, varvamp_t: float=.9, varvamp_a: float=.0, n_cpu: int=1):
    prefix = seqwin_out.with_name(seqwin_out.name + '.design')
    mkdir(prefix)

    logger.info(f'Loading Seqwin run snapshot from {seqwin_out}')
    results = load(seqwin_out / config.WORKINGDIR.results)
    markers = results.markers
    designs = pd.read_csv(seqwin_out / config.WORKINGDIR.markers_csv, index_col=False)
    n_markers = len(markers)
    n_tar = results.state.n_tar
    n_neg = results.state.n_neg
    logger.info(f'- Loaded {n_markers} signatures')

    # BLAST against core_nt
    n_hits = _blast_nt(markers, taxid, prefix, n_cpu)

    # MSA of signature regions
    logger.info('Calculating the MSA for each signature...')
    msa_prefix = prefix / MSA_DIR
    mkdir(msa_prefix)
    msa_args = zip(
        range(n_markers), 
        (m.blast for m in markers), 
        repeat(msa_prefix, n_markers)
    )
    tar_msa_paths = mp_wrapper(_msa_worker, msa_args, n_cpu, start_method=StartMethod.fork)

    # run varVAMP on each MSA
    logger.info('Running varVAMP for each MSA...')
    varvamp_prefix = prefix / VARVAMP_DIR
    mkdir(varvamp_prefix)
    varvamp_args = zip(
        repeat(varvamp_t, n_markers), 
        repeat(varvamp_a, n_markers), 
        tar_msa_paths, 
        (varvamp_prefix / Path(str(i)) for i in range(n_markers))
    )
    mp_wrapper(varvamp, varvamp_args, n_cpu, start_method=StartMethod.fork)

    # evaluate each design
    metrics = _eval_designs(varvamp_prefix, msa_prefix, n_markers, n_tar, n_neg, n_cpu)

    designs['nt_hits'] = n_hits
    designs['log10_hits'] = np.log10(designs['nt_hits'])
    designs['varVAMP_p'] = metrics[:, 0]
    designs['sensitivity'] = metrics[:, 1]
    designs['specificity'] = metrics[:, 2]
    designs['score'] = _get_scores(designs)

    designs.sort_values(
        by='score', ascending=False, inplace=True, kind='stable', na_position='last'
    )
    designs.to_csv(prefix / 'designs.csv')

    logger.info('Done')


if __name__ == '__main__':
    """
    Output: a folder named `<seqwin_out>.design`. 
    - The best varVAMP design is selected for each Seqwin signature. 
    - Some signatures might have no design. 

    Folder structure:
    - msa/: Multiple sequence alignment (MSA) of each signature. 
    - varvamp/: varVAMP designs of each signature. 
    - blast.pkl: BLAST results saved as a Pandas DataFrame (load with `pd.read_pickle()`). 
    - designs.csv: Metrics of each design, sorted by 'score'. 
        The first column is signature IDs (also used in the varvamp/ folder). 
    """
    seqwin_out = Path('seqwin-out') # Path to the seqwin output dir
    taxid = 1350 # NCBI Tax ID of ancestor; e.g., 1350 (Enterococcus) for E. faecium; excluded for core_nt BLAST
    varvamp_t = 0.9 # varVAMP --threshold (-t); default 0.9
    varvamp_a = 0 # varVAMP --n-ambig (-a); default 0
    n_cpu = 60 # Number of threads to run in parallel

    main(seqwin_out, taxid, varvamp_t, varvamp_a, n_cpu)
