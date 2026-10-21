"""Synthetic educational dataset generator for C1 IISA curriculum coverage research.

Generates realistic timestamped multi-lecture transcripts with ground-truth
prerequisite DAGs, semantic similarities, prerequisite coverage states, and
standardized coverage labels across the 6 core units of Introduction to
Information Security Analytics (IISA).
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Any

from edusync_c1.research.dataset import (
    COVERAGE_LABELS,
    REQUIRED_COLUMNS,
    validate_rows,
)
from edusync_core.settings import get_settings

# 5 IISA Course Lectures covering the user's exact 6 topics & dependencies
LECTURE_BLUEPRINTS = [
    {
        "lecture_id": "LEC_01_IISA_INFOSEC_FOUNDATIONS",
        "title": "Introduction to Information Security & Threat Landscape",
        "concepts": [
            ("Introduction to Information Security", "The CIA triad defines Confidentiality, Integrity, and Availability as the core pillars of security analytics.", []),
            ("Threat Modeling", "Threat modeling identifies attack vectors, assets, and vulnerability attack surfaces.", ["Introduction to Information Security"]),
            ("Security Policies", "Security policies define administrative and technical rules governing system protection.", ["Introduction to Information Security"]),
        ],
        "style": "aligned",
    },
    {
        "lecture_id": "LEC_02_IISA_CRYPTO_BASICS",
        "title": "Cryptography Basics & Key Systems",
        "concepts": [
            ("Cryptography Basics", "A cryptosystem transforms plaintext into ciphertext using mathematical algorithms and cryptographic keys.", ["Introduction to Information Security"]),
            ("Kerckhoffs's Principle", "Kerckhoffs's principle dictates that security must depend solely on key secrecy, never algorithm obscurity.", ["Cryptography Basics"]),
            ("Key Management", "Key lifecycle management encompasses secure key generation, escrow, rotation, and revocation.", ["Cryptography Basics"]),
        ],
        "style": "aligned",
    },
    {
        "lecture_id": "LEC_03_IISA_SYMMETRIC_HASHES",
        "title": "Symmetric Encryption & Cryptographic Hash Functions",
        "concepts": [
            ("Symmetric Encryption", "Symmetric encryption uses a single shared secret key for both encryption and decryption, exemplified by AES.", ["Cryptography Basics"]),
            ("Cryptographic Hash Functions", "Cryptographic hash functions are deterministic one-way functions offering strong collision resistance.", ["Cryptography Basics"]),
            ("Message Authentication Codes", "HMAC combines a cryptographic hash with a secret key to guarantee message authentication and integrity.", ["Cryptographic Hash Functions", "Symmetric Encryption"]),
        ],
        "style": "skip_prerequisite",  # Simulates lecturer skipping hash fundamentals and jumping directly to HMAC
    },
    {
        "lecture_id": "LEC_04_IISA_USER_AUTHENTICATION",
        "title": "User Authentication & Credential Protection",
        "concepts": [
            ("User Authentication", "User authentication verifies the claimed identity of a subject before granting access to resources.", ["Cryptographic Hash Functions"]),
            ("Salted Password Hashing", "Storing passwords requires unique random cryptographic salts combined with slow one-way hash functions like bcrypt.", ["Cryptographic Hash Functions", "User Authentication"]),
            ("Multi-Factor Authentication", "Multi-factor authentication requires two or more independent factors: knowledge, possession, or inherence.", ["User Authentication"]),
        ],
        "style": "shallow_coverage",  # Simulates lecturer skimming MFA quickly without depth
    },
    {
        "lecture_id": "LEC_05_IISA_ACCESS_CONTROL",
        "title": "Access Control Models & Authorization Governance",
        "concepts": [
            ("Access Control", "Access control governs which authenticated subjects can perform specific operations on protected objects.", ["User Authentication"]),
            ("Principle of Least Privilege", "The principle of least privilege mandates granting only the minimum permissions necessary to complete a task.", ["Access Control"]),
            ("Role-Based Access Control", "Role-Based Access Control (RBAC) assigns permissions to organizational roles rather than individual user accounts.", ["Access Control", "Principle of Least Privilege"]),
            ("Attribute-Based Access Control", "Attribute-Based Access Control (ABAC) evaluates dynamic subject, object, and environmental attributes at runtime.", ["Access Control"]),
        ],
        "style": "aligned",
    },
]

EXPANSION_TEMPLATES = {
    "deep_explanation": [
        "Let us examine how {concept} functions mathematically in security systems.",
        "A critical principle of {concept} is that {context}",
        "Notice how {concept} defends against active adversarial attacks in production environments.",
        "In modern security analytics, {concept} is essential because {context}",
        "When auditing an enterprise network, {concept} provides verifiable guarantees: {context}",
    ],
    "brief_mention": [
        "We also have {concept}, which is covered in chapter three of the security handbook.",
        "And of course {concept} exists, but we will not delve into the proof details today.",
        "Keep in mind that {concept} is relevant to our threat model here.",
    ],
    "off_topic": [
        "Please remember that your IISA lab report on network packet captures is due this Friday.",
        "Does anyone have questions about the upcoming midterm exam format?",
        "Next Tuesday we will review the incident response case study from the banking sector.",
        "Let us take a brief two-minute water break before analyzing the next cipher.",
    ],
}


def generate_synthetic_dataset(seed: int = 42) -> list[dict[str, Any]]:
    """Generate a high-quality multi-lecture dataset with calibrated labels for IISA."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for blueprint in LECTURE_BLUEPRINTS:
        lecture_id = blueprint["lecture_id"]
        style = blueprint["style"]
        concepts_info = blueprint["concepts"]

        covered_concepts: set[str] = set()
        concept_evidence_counts: dict[str, int] = {c[0]: 0 for c in concepts_info}
        timestamp = 15.0
        segment_id = 0

        # Optional intro chatter
        intro_text = rng.choice(EXPANSION_TEMPLATES["off_topic"])
        rows.append({
            "lecture_id": lecture_id,
            "segment_id": segment_id,
            "timestamp_seconds": round(timestamp, 1),
            "concept": concepts_info[0][0],
            "transcript_text": intro_text,
            "semantic_similarity": round(rng.uniform(0.05, 0.22), 3),
            "prerequisite_coverage": 0.0,
            "previous_evidence_count": 0,
            "label": "not_yet_observed",
        })
        segment_id += 1
        timestamp += rng.uniform(20.0, 35.0)

        # Ordering of presentation
        if style == "skip_prerequisite":
            # Reverse order so advanced concepts come first before prerequisites
            presented_order = list(reversed(concepts_info))
        else:
            presented_order = list(concepts_info)

        for concept_name, explanation, prereqs in presented_order:
            # Calculate prerequisite coverage at current lecture point
            if not prereqs:
                prereq_cov = 1.0
            else:
                satisfied = sum(1 for p in prereqs if p in covered_concepts)
                prereq_cov = round(satisfied / len(prereqs), 3)

            # Segment 1: Introduction of the concept
            prev_count = concept_evidence_counts[concept_name]
            intro_sim = round(rng.uniform(0.72, 0.94), 3)
            intro_text = f"Today we focus on {concept_name}. {explanation}"

            if prereq_cov >= 0.5:
                label = "introduced" if prev_count == 0 else "adequately_covered"
            else:
                # Prerequisites missing!
                label = "insufficiently_covered"

            rows.append({
                "lecture_id": lecture_id,
                "segment_id": segment_id,
                "timestamp_seconds": round(timestamp, 1),
                "concept": concept_name,
                "transcript_text": intro_text,
                "semantic_similarity": intro_sim,
                "prerequisite_coverage": prereq_cov,
                "previous_evidence_count": prev_count,
                "label": label,
            })
            concept_evidence_counts[concept_name] += 1
            segment_id += 1
            timestamp += rng.uniform(25.0, 45.0)

            # Follow-up segments depending on style
            if style == "shallow_coverage":
                # Quick brief mention only
                brief_text = rng.choice(EXPANSION_TEMPLATES["brief_mention"]).format(concept=concept_name)
                sim = round(rng.uniform(0.42, 0.65), 3)
                rows.append({
                    "lecture_id": lecture_id,
                    "segment_id": segment_id,
                    "timestamp_seconds": round(timestamp, 1),
                    "concept": concept_name,
                    "transcript_text": brief_text,
                    "semantic_similarity": sim,
                    "prerequisite_coverage": prereq_cov,
                    "previous_evidence_count": concept_evidence_counts[concept_name],
                    "label": "insufficiently_covered",
                })
                concept_evidence_counts[concept_name] += 1
                segment_id += 1
                timestamp += rng.uniform(15.0, 30.0)
            else:
                # In-depth coverage: 2-3 deep explanation segments
                num_deep = rng.randint(2, 4)
                for _ in range(num_deep):
                    tmpl = rng.choice(EXPANSION_TEMPLATES["deep_explanation"])
                    deep_text = tmpl.format(concept=concept_name, context=explanation)
                    sim = round(rng.uniform(0.78, 0.96), 3)
                    deep_prereq = 1.0 if not prereqs else round(sum(1 for p in prereqs if p in covered_concepts) / len(prereqs), 3)

                    if deep_prereq >= 0.5:
                        cov_label = "adequately_covered"
                    else:
                        cov_label = "insufficiently_covered"

                    rows.append({
                        "lecture_id": lecture_id,
                        "segment_id": segment_id,
                        "timestamp_seconds": round(timestamp, 1),
                        "concept": concept_name,
                        "transcript_text": deep_text,
                        "semantic_similarity": sim,
                        "prerequisite_coverage": deep_prereq,
                        "previous_evidence_count": concept_evidence_counts[concept_name],
                        "label": cov_label,
                    })
                    concept_evidence_counts[concept_name] += 1
                    segment_id += 1
                    timestamp += rng.uniform(30.0, 60.0)

                covered_concepts.add(concept_name)

            # Random off-topic break segment between concepts
            if rng.random() < 0.35:
                break_text = rng.choice(EXPANSION_TEMPLATES["off_topic"])
                rows.append({
                    "lecture_id": lecture_id,
                    "segment_id": segment_id,
                    "timestamp_seconds": round(timestamp, 1),
                    "concept": concept_name,
                    "transcript_text": break_text,
                    "semantic_similarity": round(rng.uniform(0.08, 0.28), 3),
                    "prerequisite_coverage": prereq_cov,
                    "previous_evidence_count": concept_evidence_counts[concept_name],
                    "label": "not_yet_observed",
                })
                segment_id += 1
                timestamp += rng.uniform(15.0, 35.0)

    # Validate all generated rows against the strict dataset schema
    return validate_rows(rows)


def save_dataset_csv(rows: list[dict[str, Any]], target_path: Path) -> None:
    """Save rows to CSV formatted according to REQUIRED_COLUMNS."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    settings = get_settings()
    default_dest = settings.data_dir / "c1" / "annotations" / "coverage_examples.csv"

    parser = argparse.ArgumentParser(description="Generate synthetic educational dataset for C1 IISA research.")
    parser.add_argument("--output", type=Path, default=default_dest, help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    rows = generate_synthetic_dataset(seed=args.seed)
    save_dataset_csv(rows, args.output)

    # Also save to package annotations directory (edusync/packages/c1/data/annotations)
    package_dest = Path(__file__).resolve().parents[3] / "data" / "annotations" / "coverage_examples.csv"
    save_dataset_csv(rows, package_dest)

    print(f"Generated {len(rows)} validated rows across {len(LECTURE_BLUEPRINTS)} IISA lectures.")
    print(f"Saved to: {args.output} and {package_dest}")


if __name__ == "__main__":
    main()
