import os
import pypandoc
import argparse

def export_md_to_pdf(md_path, pdf_path):
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    print(f"Converting {md_path} → {pdf_path} ...")
    md_text = open(md_path, "r").read()
    # Try a few sensible pdf-engine options; many systems lack pdflatex by default
    engines = ["pdflatex", "xelatex", "lualatex"]
    tried = []
    for eng in engines:
        try:
            extra = ["--standalone", f"--pdf-engine={eng}"]
            pypandoc.convert_text(md_text, to="pdf", format="md", outputfile=pdf_path, extra_args=extra)
            print("  PDF generated with engine:", eng)
            return pdf_path
        except Exception as e:
            tried.append((eng, str(e)))

    # If none of the TeX engines worked, fall back to writing HTML and show instructions
    html_path = os.path.splitext(pdf_path)[0] + ".html"
    try:
        pypandoc.convert_text(md_text, to="html", format="md", outputfile=html_path)
        print(f"PDF conversion failed for engines {', '.join(e for e,_ in tried)}.")
        print(f"Wrote HTML fallback to: {html_path}")
        print("To generate PDF you need a TeX engine (texlive) installed, or wkhtmltopdf/wkhtmltoimage.")
        print("On Debian/Ubuntu: sudo apt install texlive-latex-recommended texlive-luatex or sudo apt install wkhtmltopdf")
        return html_path
    except Exception as e:
        # give detailed troubleshooting info
        msg = "\n".join([f"engine={eng}: err={err}" for eng, err in tried])
        raise RuntimeError(f"Failed to convert to PDF or HTML. Tried engines: {msg}\nOriginal error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, help="Input Markdown file (EF_Report.md)")
    parser.add_argument("--pdf", required=True, help="Output PDF file path")
    args = parser.parse_args()

    export_md_to_pdf(args.md, args.pdf)
