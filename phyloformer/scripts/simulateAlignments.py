import argparse
import os
import subprocess

from tqdm import tqdm

SEQGEN_MODELS = ["JTT", "WAG", "PAM", "BLOSUM", "MTREV", "CPREV45", "MTART", "LG", "HIVB", "GENERAL"]

def simulate_alignments(in_dir, out_dir, seq_gen_path, model, len_seq):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    trees=[item[:-4] for item in os.listdir(in_dir) if item[-4:]=='.nwk']

    for tree in tqdm(trees):
        in_path=os.path.join(in_dir,tree+'.nwk')
        out_path= os.path.join(out_dir, tree+'.fasta')
        bash_cmd = f"{seq_gen_path} -m{model} -q -of -l {len_seq} < {in_path} > {out_path}"
        process = subprocess.Popen(bash_cmd, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to input directory containing the\
    .nwk tree files')
    parser.add_argument('-o', '--output', required=True, type=str, help='path to output directory')
    parser.add_argument('-s', '--seqgen', type=str, required=True, help='path to the seq-gen executable')
    parser.add_argument('-l', '--length', type=int, default=200, help='length of the sequences in the alignments (default: 200)')
    parser.add_argument('-m', '--model', type=str, default='PAM', choices=SEQGEN_MODELS, help=f'model of evolution (default: PAM). Allowed values: [{", ".join(SEQGEN_MODELS)}]', metavar="MODEL")
    args = parser.parse_args()

    simulate_alignments(args.input, args.output, args.seqgen, args.model, args.length)


if __name__ == '__main__':
    main()
