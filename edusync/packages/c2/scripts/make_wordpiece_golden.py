"""Write the WordPiece parity golden: expected tokens from the HF tokenizers library.

Needs the optional `tokenizers` package (not a dependency). The committed JSON lets the test suite
check the pure-Python tokenizer without it.
"""

import argparse
import json
import sys
from pathlib import Path

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = (
    ROOT.parents[1]
    / "packages"
    / "blueprint_model"
    / "artifacts"
    / "base"
    / "sentence-transformers--all-MiniLM-L6-v2"
    / "tokenizer.json"
)
DEFAULT_OUT = ROOT / "tests" / "difficulty" / "golden" / "wordpiece_pieces.json"

SENTENCES = [
    "Transparent Data Encryption (TDE) encrypts database files at rest using AES-256.",
    "The kernel schedules processes using a round-robin algorithm with a fixed time quantum.",
    "SQL injection occurs when user-supplied input is concatenated into a query string.",
    "A B-tree index reduces disk I/O for range queries in relational databases.",
    "TCP uses a three-way handshake (SYN, SYN-ACK, ACK) to establish a connection.",
    "Gradient descent minimises the loss function by iteratively updating the weights.",
    "Backpropagation computes the gradient of the cost with respect to each parameter.",
    "Overfitting happens when a model memorises noise in the training set; use regularisation.",
    "The softmax function converts logits into a probability distribution over classes.",
    "A convolutional neural network applies learnable filters to extract spatial features.",
    "Public-key cryptography relies on RSA-2048 or elliptic-curve Diffie-Hellman key exchange.",
    "A man-in-the-middle attack intercepts traffic between the client and the server.",
    "Cross-site scripting (XSS) lets attackers inject JavaScript into trusted pages.",
    "The DNS resolver caches A and AAAA records until their TTL expires.",
    "Subnetting 192.168.10.0/24 yields 254 usable host addresses.",
    "HTTP/2 multiplexes requests over a single TCP connection; TLS 1.3 reduces latency.",
    "Normalisation to 3NF removes transitive dependencies between non-key attributes.",
    "ACID transactions guarantee atomicity, consistency, isolation and durability.",
    "A deadlock requires mutual exclusion, hold-and-wait, no preemption and circular wait.",
    "Virtual memory maps pages to frames through a multi-level page table and a TLB.",
    "Semaphores and mutexes protect critical sections from race conditions.",
    "The camelCase identifier getUserName() returns the current user_name field.",
    "In Python, def fit_transform(self, X_train, y_train): returns a numpy.ndarray.",
    "for (int i = 0; i < n; i++) { sum += arr[i]; }",
    "SELECT * FROM students WHERE gpa >= 3.5 ORDER BY name DESC;",
    "The naive cafe resume uses a fianc\u00e9's fa\u00e7ade; "
    "d\u00e9j\u00e0 vu at the Z\u00fcrich caf\u00e9.",
    "Na\u00efve Bayes assumes conditional independence of features given the class label.",
    "snake_case_variable and __init__ and MAX_BUFFER_SIZE are common identifiers.",
    "Version 3.12.1 was released on 2023-12-08 at 10:45:30 UTC (RFC 3339).",
    "E = mc^2, and O(n log n) is optimal for comparison sorting; x_i != y_j.",
    "Big-O notation: O(1), O(log n), O(n^2), \u0398(n), \u03a9(n).",
    "Hash collisions are resolved via chaining or open addressing (linear probing).",
    "Docker containers share the host kernel, unlike virtual machines with hypervisors.",
    "Kubernetes orchestrates pods, services, and ingress controllers across nodes.",
    "REST APIs return JSON over HTTPS; OAuth2 issues bearer tokens (JWT).",
    "The MiniLM transformer uses multi-head self-attention and positional embeddings.",
    "Tokenisation splits text into sub-word units such as WordPiece or BPE tokens.",
    "Photosynthesis, mitochondria and ribonucleic acid are not CS terms at all.",
    "Supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis",
    "Zero-knowledge proofs let a prover convince a verifier without revealing the witness.",
    "RAID-5 stripes data with distributed parity across at least three disks.",
    "The e-mail address is john.doe+test@example.co.uk, port :8080/path?x=1&y=2#frag.",
    "Tabs\tand\nnewlines  and   multiple   spaces and non-breaking\u00a0space and zero\u200bwidth.",
    "Fullwidth \uff33\uff31\uff2c and ligature \ufb01nal \u00dfeta, Turkish \u0130stanbul, "
    "Greek \u03a3\u03af\u03c3\u03c5\u03c6\u03bf\u03c2, "
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 mixed.",
    "Emoji \U0001f642 test \u2014 em-dash, \u201cquotes\u201d, \u2018single\u2019, "
    "ellipsis\u2026 and \u00bd fraction, \u00b2 superscript.",
    "AI-powered semi-supervised learning isn't the same as self-supervised (it's different).",
    "Control chars \x01\x02 and the replacement char \ufffd should be dropped, "
    "soft\u00adhyphen too.",
    "",
    "   ",
    "a" * 99,
    "a" * 100,
    "a" * 101,
    "a" * 150 + " short " + "b" * 100 + " " + "c" * 99,
]


def main(argv: list[str] | None = None) -> int:
    """Write the golden JSON; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.tokenizer.is_file():
        print(f"tokenizer file not found: {args.tokenizer}", file=sys.stderr)
        return 2
    from tokenizers import Tokenizer  # optional dependency, imported only when needed

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    cases = [
        {"text": text, "tokens": tokenizer.encode(text, add_special_tokens=False).tokens}
        for text in SENTENCES
    ]
    payload = {"revision": REVISION, "cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
