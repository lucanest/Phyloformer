import os, subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, help='path to input directory containing the\
    .nwk tree files')
    parser.add_argument('--o', type=str, help='path to output directory')
    parser.add_argument('--sg', type=str, help='path to seq-gen')
    parser.add_argument('--l', type=str, default='200', help='lenght of the sequences in the alignments')
    parser.add_argument('--m', type=str, default='PAM', help='model of evolution')
    args = parser.parse_args()

    in_dir = args.i
    out_dir = args.o
    len_seq = args.l
    seq_gen_path = args.sg
    model = args.m

    trees=[item[:-4] for item in os.listdir(in_dir) if item[-4:]=='.nwk']

    for tree in trees:
        in_path=os.path.join(in_dir,tree+'.nwk')
        out_path= out_dir+tree+'.fasta'
        bash_cmd = seq_gen_path + '/seq-gen -m'+model+' -q -of -l '+ len_seq +' < '+ in_path + ' > ' + out_path
        process = subprocess.Popen(bash_cmd, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()


if __name__ == '__main__':
    main()
