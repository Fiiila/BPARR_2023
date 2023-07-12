from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

if __name__=="__main__":
    directory = Path("./BPARR_latex/")

    with ZipFile("BPARR_latex/BPARR_latex_sync/BPARR_latex.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        # compress single files in directory to exclude temp files
        file_list = ["Jasek_BPARR_2023.tex",
                     "Jasek_BPARR_2023.pdf",
                     "fasthesis.cls",
                     "sources.bib",
                     "zadani.pdf",
                     "latexmkrc"
                     ]
        for cust_file in file_list:
            cust_file = directory.joinpath(cust_file)
            archive.write(cust_file, arcname=cust_file.relative_to(directory))
        directory_list = ["texmf",
                          "img",
                          "Graphics"
                          ]
        for sub_directory in directory_list:
            cust_dir = directory.joinpath(sub_directory)
            for file_path in cust_dir.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(directory))