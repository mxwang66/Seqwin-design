__author__ = 'Michael X. Wang'

import logging
from enum import Enum
from pathlib import Path
from itertools import product
from types import MappingProxyType
from collections.abc import Iterable
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner, Alignment, substitution_matrices
from plotly import graph_objects as go

from seqwin.utils import StartMethod, mp_wrapper, get_chunks

logger = logging.getLogger(__name__)
rng = np.random.default_rng(seed=0)

# Numpy array types
FloatArr = NDArray[np.float64]
StrArr = NDArray[np.str_]
BoolArr = NDArray[np.bool_]

# Define ambiguous bases (IUPAC codes)
_DNA = ('A', 'T', 'C', 'G', '-') # defines the columns in _AMB2FREQ
_AMB2FREQ: dict[str, FloatArr] = {
                  # A,   T,   C,   G,   -
    'A': np.array([1.0, 0.0, 0.0, 0.0, 0.0]), 
    'T': np.array([0.0, 1.0, 0.0, 0.0, 0.0]), 
    'C': np.array([0.0, 0.0, 1.0, 0.0, 0.0]), 
    'G': np.array([0.0, 0.0, 0.0, 1.0, 0.0]), 
    '-': np.array([0.0, 0.0, 0.0, 0.0, 1.0]), 
    'U': np.array([0.0, 1.0, 0.0, 0.0, 0.0]), 
    'N': np.array([1/4, 1/4, 1/4, 1/4, 0.0]), 
    'R': np.array([1/2, 0.0, 0.0, 1/2, 0.0]), 
    'Y': np.array([0.0, 1/2, 1/2, 0.0, 0.0]), 
    'K': np.array([0.0, 1/2, 0.0, 1/2, 0.0]), 
    'M': np.array([1/2, 0.0, 1/2, 0.0, 0.0]), 
    'S': np.array([0.0, 0.0, 1/2, 1/2, 0.0]), 
    'W': np.array([1/2, 1/2, 0.0, 0.0, 0.0]), 
    'B': np.array([0.0, 1/3, 1/3, 1/3, 0.0]), 
    'D': np.array([1/3, 1/3, 0.0, 1/3, 0.0]), 
    'H': np.array([1/3, 1/3, 1/3, 0.0, 0.0]), 
    'V': np.array([1/3, 0.0, 1/3, 1/3, 0.0])
}
_AMB2REG = MappingProxyType({k: frozenset(v) for k, v in {
    'A': 'A', 
    'C': 'C', 
    'G': 'G', 
    'T': 'T', 
    'U': 'T', 
    'R': 'AG', 
    'Y': 'CT', 
    'S': 'GC', 
    'W': 'AT', 
    'K': 'GT', 
    'M': 'AC', 
    'B': 'CGT', 
    'D': 'AGT', 
    'H': 'ACT', 
    'V': 'ACG', 
    'N': 'ACGT', 
    '-': '-'
}.items()})
_ALPHABET = set(_AMB2FREQ)

# create substitution matrix for pairwise aligners
_SUBMTX = substitution_matrices.Array(alphabet=''.join(_ALPHABET), dims=2)
_SUBMTX[:] = -1 # set mismatch_score = -1
for b1, b2 in (
    ('A', 'A'), ('A', 'N'), ('A', 'R'), ('A', 'M'), ('A', 'W'), ('A', 'D'), ('A', 'H'), ('A', 'V'), 
    ('T', 'T'), ('T', 'N'), ('T', 'Y'), ('T', 'K'), ('T', 'W'), ('T', 'B'), ('T', 'D'), ('T', 'H'), 
    ('C', 'C'), ('C', 'N'), ('C', 'Y'), ('C', 'M'), ('C', 'S'), ('C', 'B'), ('C', 'H'), ('C', 'V'), 
    ('G', 'G'), ('G', 'N'), ('G', 'R'), ('G', 'K'), ('G', 'S'), ('G', 'B'), ('G', 'D'), ('G', 'V'), 
    ('U', 'U'), ('U', 'T'), ('U', 'N'), ('U', 'Y'), ('U', 'K'), ('U', 'W'), ('U', 'B'), ('U', 'D'), ('U', 'H')
):
    _SUBMTX[b1, b2] = 1  # set match_score = -1
    _SUBMTX[b2, b1] = 1

# create pairwise aligner
_ALIGNER_LOCAL = PairwiseAligner(
    # match_score and mismatch_score are defined in substitution_matrix
    mode='local', 
    open_gap_score=-2, 
    extend_gap_score=-1, 
    substitution_matrix=_SUBMTX
)
_ALIGNER_GLOBAL = PairwiseAligner(
    mode='global', 
    open_gap_score=-2, 
    extend_gap_score=-1, 
    substitution_matrix=_SUBMTX
)


class Strand(str, Enum):
    """Nucleotide strands."""
    fw = '+'
    rv = '-'


def _msa_to_matrix(msa: Iterable[str]) -> StrArr:
   """Convert an MSA into a Numpy matrix. 

    Args:
        msa (Iterable[str]): An Iterable of the sequences in an MSA. All sequences should have the same length. 
    
    Returns:
        StrArr: A 2-D Numpy array with the same shape of the MSA. 
    """
   return np.array(list(
       list(s) for s in msa
    ))


def _count_bases(seq: Iterable[str]) -> FloatArr:
    """Calculate the frequencies of 'A', 'T', 'C', 'G', '-'. 
    When there is an ambiguous base, use the frequencies defined in `_AMB2FREQ`. 

    Args:
        seq (Iterable): An Iterable of nucleotide bases (see `_ALPHABET`). 
    
    Returns:
        FloatArr: A 1-D Numpy array of the frequencies of 'A', 'T', 'C', 'G' and '-', respectively. 
    """
    try:
        c = sum(_AMB2FREQ[b] for b in seq)
    except KeyError:
        # from None: to suppress "During handling of the above exception, another exception occurred"
        # so that only the ValueError is shown to the user
        # otherwise, the KeyError is also shown, which is redundant
        raise ValueError(
            f'{set(seq)-_ALPHABET} does not belong to the nucleotide alphabet, but found in input sequence.'
        ) from None
    return c


def _check_seq(seq: str) -> str:
    """Sanity check for a nucleotide sequence. 

    Args:
        seq (str): Input sequence (ambiguous bases should comply with IUPAC codes). 
    
    Returns:
        str: Input sequence in upper cases. 
    """
    seq = seq.upper()
    if not set(seq).issubset(_ALPHABET):
        raise ValueError(f'{set(seq)-_ALPHABET} does not belong to IUPAC codes, but found in the input sequence.')
    elif len(seq) < 2:
        raise ValueError(f'Input sequence is shorter than 2nt.')
    return seq


class Oligo(str):
    """The Oligo class, inherited from `str`. 

    Attributes:
        expanded (set[str]): A set of all possible oligos without any ambiguous bases. 
    """
    __slots__ = (
        '__dict__' # cached_property needs __dict__ to function
    )

    def __new__(cls, seq: str):
        """Create an Oligo instance. 

        Args:
            seq (str): The oligo sequence (ambiguous bases should comply with the IUPAC codes). 
        """
        return super().__new__(cls, _check_seq(seq))

    def __setattr__(self, name, value):
        raise AttributeError('Oligo instances are immutable.')

    @cached_property
    def expanded(self) -> frozenset[str]:
        """A set of all possible oligos without any ambiguous bases. 
        """
        expanded = (_AMB2REG[base] for base in self)
        return frozenset(''.join(p) for p in product(*expanded))


class AttachedOligo(Oligo):
    """An oligo attached to a template, inherited from the Oligo class. 

    Attributes:
        alignment (Alignment): Local alignment between the oligo and the template. 
        start (int): Alignment start in the template. 
        stop (int): Alignment stop in the template. 
        strand (Strand): Aligned strand of the template. 
        formatted (str): Formatted alignment as a string. 
    """
    __slots__ = (
        'alignment', 'start', 'stop', 'strand'
    )
    alignment: Alignment
    start: int
    stop: int
    strand: Strand

    def __new__(cls, oligo: str, template: str):
        # initialize from the Oligo class
        return super().__new__(cls, oligo)

    def __init__(self, oligo: str, template: str) -> None:
        """Create an AttachedOligo instance by attaching an oligo to a template. 
        Ambiguous bases should comply with the IUPAC codes. 

        Args:
            oligo (str): The oligo sequence. 
            template (str): The template sequence. 
        """
        # use the oligo sequence initialized in __new__
        oligo = self

        # find where the oligo is aligned on the template (local alignment)
        # oligo must be the target (first arg), so that its sequence is unchanged
        template = _check_seq(template)
        aln_fw = _ALIGNER_LOCAL.align(oligo, template)[0] # align to forward strand
        aln_rv = _ALIGNER_LOCAL.align(oligo, template, strand=Strand.rv)[0] # align to reverse strand
        aln_local = max(aln_fw, aln_rv, key=lambda x: x.score) # choose the best strand

        # oligo (target) might not aligned head to tail
        left_overhang = int(aln_local.coordinates[0][0])
        right_overhang = len(oligo) - int(aln_local.coordinates[0][-1])

        # get alignment start/stop on the template
        c1 = int(aln_local.coordinates[1][0])
        c2 = int(aln_local.coordinates[1][-1])
        if c1 <= c2:
            # forward strand
            strand = Strand.fw
            start = max(c1 - left_overhang, 0)
            stop = min(c2 + right_overhang, len(template))
            template_slice = Seq(aln_local.query[start:stop])
        else:
            # reverse strand
            strand = Strand.rv
            start = max(c2 - right_overhang, 0)
            stop = min(c1 + left_overhang, len(template))
            template_slice = Seq(aln_local.query[start:stop]).reverse_complement()

        # since local alignment might have overhangs, align again with global alignment
        alignment = _ALIGNER_GLOBAL.align(oligo, template_slice)[0]

        object.__setattr__(self, 'alignment', alignment)
        object.__setattr__(self, 'start', start)
        object.__setattr__(self, 'stop', stop)
        object.__setattr__(self, 'strand', strand)

    def __str__(self):
        return self.formatted

    @cached_property # __dict__ is defined in parent class
    def formatted(self) -> str:
        """Format the alignment into a pretty string to be printed. 
        """
        alignment = self.alignment
        start = self.start
        stop = self.stop
        strand = self.strand

        # get the formatted string from biopython
        oligo, middle_str, template, _ = alignment._format_unicode().split('\n')

        # generate the formatted string
        if strand == Strand.rv:
            start, stop = stop, start
        padding_spaces = ' ' * (len(str(start))-1) # align the lines for printing
        line1 = f'{padding_spaces}0 {oligo} {len(oligo)}\n'
        line2 = f'{padding_spaces}  {middle_str}\n'
        line3 =           f'{start} {template} {stop}\n'
        return f'{line1}{line2}{line3}'


class MSA(object):
    """Load, process and plot multiple sequence alignment (MSA). 

    Attributes:
        msa (StrArr): A 2-D Numpy array of the MSA. 
        consensus (str): Consensus sequence without any gaps ('-'). 
        oligos (dict[str, tuple[AttachedOligo, BoolArr]]): Oligos attached to the MSA, 
            with a boolean array indicating perfectly matched sequences. 
    """
    __slots__ = (
        'msa', 'consensus', 'oligos', '_c_arr', '_c2m', '_m2c', '_vote'
    )
    msa: StrArr
    consensus: str
    oligos: dict[str, tuple[AttachedOligo, BoolArr]]
    _c_arr: StrArr # consensus with gaps
    _c2m: dict[int, int] # a mapping of positions in the consensus to MSA column indices
    _m2c: dict[int, int] # a mapping of MSA column indices to positions in the consensus
    _vote: FloatArr # the number of each base for each MSA column

    def __setattr__(self, name, value):
        raise AttributeError('MSA instances are immutable.')

    def __init__(self, path: Path, n_cpu: int=1) -> None:
        """Load MSA from a FASTA file. 

        Args:
            path (Path): Path to the MSA file in FASTA format. 
            n_cpu (int): Number of threads to use. [1]
        """
        msa = list(
            str(record.seq).upper() for record in SeqIO.parse(path, 'fasta')
        )
        # convert to numpy matrix
        if len(msa) <= n_cpu or n_cpu == 1:
            msa = _msa_to_matrix(msa)
        else:
            msa: StrArr = np.concatenate(mp_wrapper(
                _msa_to_matrix, get_chunks(msa, n_cpu), n_cpu, 
                starmap=False, start_method=StartMethod.fork
            ))

        # Bio.Align class can also read an MSA, and convert to matrix, but slower
        # msa = Align.read(path, 'fasta')
        # matrix = np.array(msa, dtype='U')

        consensus, c_arr, c2m, m2c, vote = MSA.__get_consensus(msa, msa.shape[1], n_cpu)

        object.__setattr__(self, 'msa', msa)
        object.__setattr__(self, 'consensus', consensus)
        object.__setattr__(self, 'oligos', dict())
        object.__setattr__(self, '_c_arr', c_arr)
        object.__setattr__(self, '_c2m', c2m)
        object.__setattr__(self, '_m2c', m2c)
        object.__setattr__(self, '_vote', vote)

    @staticmethod
    def __get_consensus(
        msa: np.array, 
        col: int, 
        n_cpu: int=1
    ) -> tuple[
        str, 
        StrArr, 
        dict[int, int], 
        dict[int, int], 
        FloatArr
    ]:
        """Calculate the concensus sequence. 
        Args:
            msa (np.array): MSA as a Numpy matrix. 
            col (int): Number MSA columns. 
            n_cpu (int): Number of threads to use. [1]
        
        Returns:
            tuple: A tuple containing
                str: See `MSA.consensus`. 
                StrArr: See `MSA._c_arr`. 
                dict[int, int]: See `MSA._c2m`. 
                dict[int, int]: See `MSA._m2c`. 
                FloatArr: See `MSA._vote`. 
        """
        # count the number of each base for each MSA column
        # returns a matrix with 5 rows (bases) and col columns
        vote = np.array(mp_wrapper(
            _count_bases, 
            (msa[:, i] for i in range(col)), 
            n_cpu, 
            starmap=False, 
            start_method=StartMethod.fork
        )).T # transpose the matrix, so that it has 5 rows (ATCG-) and N columns

        # find the most frequent base for each column (consensus array, with gaps)
        c_arr = np.array(
            list(_DNA[i] for i in np.argmax(vote, axis=0))
        )

        # get the gapless consensus string, and a mapping of its positions to MSA columns, 
        # so that when a sequence is aligned to the gapless consensus, we can also know where it aligns to the MSA
        consensus = list()
        c2m = dict() # position in c_arr -> MSA column index
        m2c = dict() # MSA column index -> position in c_arr
        loc = 0 # position on consensus
        for i_col, b in enumerate(c_arr):
            m2c[i_col] = loc
            if b != '-':
                consensus.append(b)
                c2m[loc] = i_col
                loc += 1
        c2m[loc] = i_col + 1

        return ''.join(consensus), c_arr, c2m, m2c, vote

    def attach_oligo(self, seq: str, name: str | None=None) -> BoolArr:
        """Find the aligned position of an oligo on the MSA, and find sequences with a perfect match. 

        Args:
            seq (str): The oligo sequence (ambiguous bases should comply with the IUPAC codes). 
            name (str | None): Name of the oligo. If None, use 'Oligo-X' where X is the number of oligos attached to this MSA. [None]
        
        Returns:
            BoolArr: A boolean array indicating perfectly matched sequences in the MSA. 
        """
        msa = self.msa
        consensus = self.consensus
        c2m = self._c2m
        oligos = self.oligos

        if name is None:
            name = f'Oligo-{len(oligos)}'
        olg = AttachedOligo(seq, consensus)

        # fetch MSA columns of the oligo
        msa_slice = msa[:, 
            c2m[olg.start]: c2m[olg.stop]
        ]
        msa_slice = (''.join(s).replace('-', '') for s in msa_slice)
        if olg.strand == Strand.rv:
            olg_expanded = frozenset(
                str(Seq(o).reverse_complement()) for o in olg.expanded
            )
        else:
            olg_expanded = olg.expanded

        # find sequences with a perfect match
        matches = np.array(list(
            True if s in olg_expanded else False for s in msa_slice
        ))
        oligos[name] = (olg, matches)

        object.__setattr__(self, 'oligos', oligos)
        return matches

    def plot(self, save_path: Path|None=None) -> None:
        row, col = self.msa.shape
        oligos = self.oligos
        consensus = self.consensus
        c2m = self._c2m
        vote = self._vote

        fig = go.Figure()
        # plot MSA
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'lightgray']
        x = np.array(range(col + 1))
        # x axis: [0, 1, 1, 2, 2, 3, 3, 4, 4, ..., col-1, col-1, col]
        x_plot = np.repeat(x, 2)[1:-1]
        for i, base in enumerate(_DNA):
            freq_plot = np.repeat(vote[i, :], 2)
            fig.add_scatter(
                x=x_plot, 
                y=freq_plot, 
                hoverinfo='skip', 
                name=base, 
                mode='lines', 
                line=dict(width=0, color=colors[i]), 
                stackgroup='one'
            )
        
        # plot oligos
        all_olg_x = list()
        all_olg_y = list()
        all_hover_text = list()
        all_mismatch_x = list()
        all_mismatch_y = list()
        # coordinates on y-axis are determined randomly
        olg_y_prev = row / 2
        y_min, y_max, y_dist = 0.25 * row, 0.75 * row, 0.2 * row
        for olg, matches in sorted(
            # sort oligos by positions, so that the current oligo won't be placed to close to the previous one
            oligos.values(), key=lambda v: min(v[0].start, v[0].stop)
        ):
            if olg.strand == Strand.fw:
                olg_x = [
                    c2m[olg.start], 
                    c2m[olg.stop], 
                    None # add None so that oligos will not be connected
                ]
            else:
                olg_x = [
                    c2m[olg.stop], 
                    c2m[olg.start], 
                    None
                ]
            all_olg_x.extend(olg_x)

            # place randomly on y-axis
            olg_y = rng.uniform(y_min, y_max)
            while abs(olg_y - olg_y_prev) < y_dist:
                olg_y = rng.uniform(y_min, y_max)
            olg_y_prev = olg_y
            all_olg_y.extend([olg_y, olg_y, None])

            # hover text should be html style
            matches = sum(matches) # number of perfectly matched sequences in the MSA
            hover_text = f'sensitivity: {100*matches/row:.2f}% ({matches}/{row})\noligo/consensus:\n{olg.formatted}'
            hover_text = hover_text.replace('\n', '<br>')
            all_hover_text.extend([hover_text, hover_text, None])

            # lable mismatches and gaps (gaps pending)
            _, middle_str, template, _ = olg.alignment._format_unicode().split('\n')
            if olg.strand == Strand.rv:
                middle_str = middle_str[::-1]
                template = template[::-1]
            curr_consensus_pos = olg.start
            mismatch_x = list()
            for i, s in enumerate(middle_str):
                if s == '|':
                    curr_consensus_pos += 1
                    continue
                elif s == '.':
                    mismatch_x.append(c2m[curr_consensus_pos]+0.5)
                    curr_consensus_pos += 1
                else: # gap
                    if template[i] == '-':
                        # insertion (place the label between the two bases)
                        mismatch_x.append(c2m[curr_consensus_pos])
                    else:
                        # deletion
                        mismatch_x.append(c2m[curr_consensus_pos]+0.5)
                        curr_consensus_pos += 1
            mismatch_y = [olg_y]*len(mismatch_x)
            all_mismatch_x.extend(mismatch_x)
            all_mismatch_y.extend(mismatch_y)

        fig.add_scatter(
            x=all_olg_x, 
            y=all_olg_y, 
            showlegend=False, 
            name='', 
            marker=dict(size=12, symbol='arrow-bar-up', angleref='previous'), 
            line=dict(width=5, color='black'), 
            connectgaps=False, 
            hovertemplate='%{text}', 
            text=all_hover_text
        )
        fig.add_scatter(
            x=all_mismatch_x, 
            y=all_mismatch_y, 
            name='Substitution or Indel', 
            mode='markers', 
            marker=dict(
                size=12, 
                symbol='x-thin', 
                line=dict(width=3, color='blueviolet')
            ), 
            hoverinfo='skip'
        )

        # title and axis
        fig.update_layout(
            # hover label
            hoverlabel=dict(
                bgcolor='white', 
                font_family='Overpass, monospace' # use mono font to show alignment correctly
            ), 

            # title
            title=dict(
                text=f'MSA of {row} sequences ({col} columns); length of consensus: {len(consensus)}', 
                #text=f'MSA of {self.row} H1N1 HA sequences (GISAID 04/25/24-10/10/24); length of consensus: {len(self.consensus)}', 
                xanchor='left', 
                x=0.06, 
                # yanchor='bottom', 
                # y=0.8, 
                font=dict(size=18)
            ), 

            # legend
            legend=dict(
                orientation='h', 
                xanchor='right', 
                x=1, 
                yanchor='bottom', 
                y=1, 
                traceorder='normal' # A, T, C, G are shown left to right
            ), 

            # axis
            xaxis=dict(
                # setting customized tick lables will show all lables at all times, zoom-in or not
                # tickmode='array', 
                # tickvals=list(self._msa2consensus), 
                # ticktext=list(self._msa2consensus.values()), 
                tickfont=dict(
                    size=16, 
                ), 
                showgrid=False, 
                rangeslider=dict(
                    visible=True
                )
            ), 
            yaxis=dict(
                range=[0, row], 
                tickfont=dict(
                    size=16, 
                ), 
                showgrid=False
            ), 

            # axis title
            xaxis_title='MSA position (oligo coordinates are 0-start, half-open)', 
            yaxis_title='# bases', 

            # global font
            font=dict(
                size=18, 
                family='Arial'
            )
        )
        if save_path is None:
            fig.show()
        else:
            if save_path.suffix != '.html':
                save_path = save_path.with_suffix('.html')
            fig.write_html(f'{save_path}')
