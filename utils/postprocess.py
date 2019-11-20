import codecs
import argparse

def del_repeat(prediction, output):
    out_file = codecs.open(output, 'w', 'utf-8')
    with codecs.open(prediction, 'r', 'utf-8') as f:
        while True:
            pline = f.readline()
            if pline == '':
                break
            pphrase = pline.strip().split(' ')[:-1]
            ophrase = []
            for p in pphrase:
                if p in ophrase:
                    continue
                else:
                    ophrase.append(p)
            out_file.write(" ".join(ophrase) + '\n')

def del_last(prediction, output):
    out_file = codecs.open(output, 'w', 'utf-8')
    with codecs.open(prediction, 'r', 'utf-8') as f:
        while True:
            pline = f.readline()
            if pline == '':
                break
            pphrase = pline.strip().split(' ')[:-1]
            out_file.write(" ".join(pphrase) + '\n')


def to_lower(reference, output):
    out_file = codecs.open(output, 'w', 'utf-8')
    with codecs.open(reference, 'r', 'utf-8') as f:
        while True:
            pline = f.readline()
            if pline=="":
                break
            pline = pline.lower()
            out_file.write(pline)

def main():
    parser = argparse.ArgumentParser(description='postprocess.py')
    parser.add_argument('-input', required=True,
                        help='Path of output file to be processed')
    parser.add_argument('-output', required=True,
                        help='Path to save')
                        
    opt = parser.parse_args()
    pred_file = opt.input
    output = opt.output 
    # ref_file = '../raw_data/test.msg'
    # reference = '../exp/reference.txt'
    # del_repeat(pred_file, output)
    # del_last(pred_file, output)
    to_lower(pred_file, output)


if __name__ == '__main__':
    main()
    