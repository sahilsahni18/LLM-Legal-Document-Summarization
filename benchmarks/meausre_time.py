import time
from src.ocr_pipeline import ocr_pdf
from src.clause_detector import inference

def time_pipeline(pdf_path):
    start = time.time()
    text = ocr_pdf(pdf_path)
    scores = inference([text], model_dir="clause_model")
    return time.time() - start

if __name__ == "__main__":
    import glob
    times = []
    for pdf in glob.glob("data/test_pdfs/*.pdf"):
        t = time_pipeline(pdf)
        times.append(t)
        print(f"{pdf}: {t:.2f}s")
    avg = sum(times)/len(times)
    print(f"Average end-to-end time: {avg:.2f}s")
