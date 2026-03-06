## Seqwin design pipeline

> [!NOTE]
> **This pipeline is still being tested. Please always validate the output designs manually.**

## Install with Bioconda

**Prerequisites**
- Linux, macOS, or Windows via [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- x64 or ARM64
- conda (install with [miniforge](https://github.com/conda-forge/miniforge#install) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions))

**1. Create a new Conda environment "pcr" and install dependencies via Bioconda**
```bash
conda create -n pcr seqwin varvamp mafft \
  --channel conda-forge \
  --channel bioconda \
  --strict-channel-priority
```

**2. Activate the environment and verify the install**
```bash
conda activate pcr
seqwin --help
varvamp --help
```

**3. Set up a local `core_nt` BLAST database**

Check with your administrator if this has been done already. 
- Download `core_nt` to your home directory (this might take a while). 
  ```bash
  conda activate pcr
  mkdir -p ~/BLASTDB && cd ~/BLASTDB
  update_blastdb.pl --decompress taxdb core_nt && cd ~
  ```
- Add this line to your `.bashrc` and restart the terminal. 
  ```bash
  export BLASTDB="$HOME/BLASTDB"
  ```

**4. Clone this repository**
```bash
git clone https://github.com/mxwang66/Seqwin-design.git
```

## Usage

**1. Run Seqwin**
- Do NOT use `--no-blast`. Use `--no-mash` if the number of genomes is large (>5,000). 
- The ideal number of output signatures is 200-300. Default stringency (`-s <int>`) is 5. 
- Decrease stringency (e.g., `-s 4`) if the number of output signatures is too low (or if no signatures are found). 
- Increase stringency (e.g., `-s 6`) if the number of output signatures is too high. 

Check the [GitHub repository](https://github.com/treangenlab/Seqwin) for more details. 

**2. Run `design.py`**
- Use the instructions at the end of `design.py` and set up parameters (after `if __name__ == '__main__':`). 
- Run the script in the terminal. 
  ```bash
  conda activate pcr
  cd Seqwin-design
  python design.py
  ```