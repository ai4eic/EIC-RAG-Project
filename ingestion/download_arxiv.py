import argparse, arxiv, os
from joblib import Parallel, delayed

def download_source(arxiv_id, output_folder, cwd):
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    success = False
    try:
        pth = os.path.join(output_folder, arxiv_id)
        if(not os.path.exists(pth)):
            os.makedirs(pth)
        paper.download_source(dirpath = pth, filename = "source.tar.gz")
        paper.download_pdf(dirpath = pth, filename = arxiv_id + ".pdf")
        os.chdir(pth)
        os.system("tar -zxvf source.tar.gz")
        print("Downloaded arxiv_id: ", arxiv_id)
        success = True
    except Exception as e:
        print(e)
        print("Failed to download arxiv_id: ", arxiv_id)
    os.chdir(cwd)
    return success
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input_file', type=str, required=True)
    argparser.add_argument('-o', '--output_folder', type=str, required=True, default = "ARXIV_SOURCES")
    argparser.add_argument('-n', '--nthreads', type=int, default=-1)
    args = argparser.parse_args()
    input_file = args.input_file
    output_folder = args.output_folder
    nthreads = args.nthreads
    cwd = os.getcwd()
    if(not os.path.exists(output_folder)):
        os.makedirs(output_folder)
    arxiv_ids = [i.strip() for i in open(input_file, 'r').readlines()]
    if(nthreads == -1):
        [download_source(arxiv_id, output_folder, cwd) for arxiv_id in arxiv_ids]
    else:
        Parallel(n_jobs=nthreads)(delayed(download_source)(arxiv_id, output_folder, cwd) for arxiv_id in arxiv_ids)