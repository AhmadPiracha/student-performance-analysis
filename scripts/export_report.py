import os
import pypandoc
import argparse

def export_md_to_pdf(md_path, pdf_path):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    print(f"Converting {md_path} → {pdf_path} ...")
    output = pypandoc.convert_text(
        open(md_path, "r").read(),
        to="pdf",
        format="md",
        outputfile=pdf_path,
        extra_args=["--standalone"]
    )
    print("  PDF generated:", pdf_path)
    return pdf_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, help="Input Markdown file (EF_Report.md)")
    parser.add_argument("--pdf", required=True, help="Output PDF file path")
    args = parser.parse_args()

    export_md_to_pdf(args.md, args.pdf)
