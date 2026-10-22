"""Generate sample university lecture PDFs across multiple courses (IISA, ML, OS)
to demonstrate that EduTwin-AI Component C1 is domain-agnostic and works on ANY lecture PDF.
"""

from io import BytesIO
from pathlib import Path
from pypdf import PdfWriter


def make_multipage_pdf(pages_lines: list[list[str]]) -> bytes:
    """Build a multi-page PDF with standard text objects."""
    writer = PdfWriter()

    for lines in pages_lines:
        stream = "BT /F1 14 Tf 72 720 Td 18 TL " + " T* ".join(f"({line}) Tj" for line in lines) + " ET"
        objects = [
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        out = bytearray(b"%PDF-1.4\n")
        offsets = []
        for number, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
        xref = len(out)
        out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode()
        out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()

        reader = PdfWriter()
        from pypdf import PdfReader
        page_reader = PdfReader(BytesIO(bytes(out)))
        writer.add_page(page_reader.pages[0])

    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def main():
    repo_root = Path(__file__).resolve().parent.parent
    samples_dir = repo_root / "edusync" / "packages" / "c1" / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Course 1: IISA (Introduction to Information Security Analytics)
    iisa_pdf = make_multipage_pdf([
        [
            "Course: Introduction to Information Security Analytics",
            "Unit 1: Introduction to Information Security",
            "The CIA Triad defines Confidentiality, Integrity, and Availability as the core pillars of security analytics.",
            "Information security policies establish baseline organizational rules for protection."
        ],
        [
            "Unit 2: Cryptography Basics",
            "Cryptography Basics transforms plaintext into ciphertext using mathematical keys.",
            "Kerckhoffs's principle dictates that security must depend on key secrecy, not obscurity."
        ],
        [
            "Unit 3: Symmetric Encryption & Cryptographic Hash Functions",
            "Symmetric Encryption uses a shared private key for fast block-cipher encryption.",
            "Cryptographic Hash Functions provide deterministic one-way data integrity verification."
        ],
        [
            "Unit 4: User Authentication",
            "User Authentication verifies digital identities using salted password hashing.",
            "Multi-Factor Authentication enforces layered verification tokens."
        ],
        [
            "Unit 5: Access Control & Security Governance",
            "Access Control regulates operations permitted to authenticated identities.",
            "Role-Based Access Control assigns permissions to organizational roles based on the principle of least privilege."
        ]
    ])
    (samples_dir / "IISA_Information_Security.pdf").write_bytes(iisa_pdf)
    print(f"[+] Generated: {samples_dir / 'IISA_Information_Security.pdf'}")

    # 2. Course 2: Machine Learning Foundations
    ml_pdf = make_multipage_pdf([
        [
            "Course: Machine Learning Foundations",
            "Unit 1: Introduction to Machine Learning",
            "Supervised learning trains predictive models from labeled training examples.",
            "Feature engineering transforms raw observations into numerical representations."
        ],
        [
            "Unit 2: Optimization with Gradient Descent",
            "Gradient Descent is an optimization algorithm that minimizes the Loss Function.",
            "The Learning Rate determines the step size taken along the negative gradient vector."
        ],
        [
            "Unit 3: Deep Neural Networks",
            "Artificial Neural Networks compose non-linear activation layers to model complex representations.",
            "Backpropagation computes partial derivatives using the mathematical chain rule to update synaptic weights."
        ]
    ])
    (samples_dir / "Machine_Learning_Foundations.pdf").write_bytes(ml_pdf)
    print(f"[+] Generated: {samples_dir / 'Machine_Learning_Foundations.pdf'}")

    # 3. Course 3: Operating Systems & Concurrency
    os_pdf = make_multipage_pdf([
        [
            "Course: Operating Systems Architecture",
            "Unit 1: Process Management & CPU Scheduling",
            "An operating system manages concurrent Process Scheduling using priority queues.",
            "Context switching saves the execution state of a thread to resume it later."
        ],
        [
            "Unit 2: Concurrency & Synchronization",
            "Thread Synchronization prevents race conditions when accessing shared memory regions.",
            "Mutex Locks ensure mutual exclusion within the critical section of concurrent code."
        ],
        [
            "Unit 3: Deadlock Detection & Prevention",
            "Deadlock occurs when processes are blocked waiting for circular resource allocation.",
            "The Banker's Algorithm ensures safe system states by evaluating maximum resource demands."
        ]
    ])
    (samples_dir / "Operating_Systems_Concurrency.pdf").write_bytes(os_pdf)
    print(f"[+] Generated: {samples_dir / 'Operating_Systems_Concurrency.pdf'}")


if __name__ == "__main__":
    main()
