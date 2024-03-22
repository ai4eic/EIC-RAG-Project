import re, os, sys, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a LateX file into multiple files.')
    parser.add_argument('input', metavar='input', type=str, nargs=1,
                        help='input file')
    parser.add_argument('output', metavar='output', type=str, nargs=1,
                        help='output directory')
    parser.add_argument('splitter', metavar='splitter', type=str, nargs=1,
                        help='splitter')
    args = parser.parse_args()
    input = args.input[0]
    output = args.output[0]
    splitter = args.splitter[0]

    if not os.path.exists(output):
        os.makedirs(output)

    with open(input, 'r') as f:
        content = f.read()
        splitted = re.split(splitter, content)
        for i, s in enumerate(splitted):
            with open(os.path.join(output, str(i) + ".tex"), 'w') as f:
                f.write(s)