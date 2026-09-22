"""MIT AI Risk Repository — Domain Taxonomy (Slattery et al., 2024).

Maps each of the 24 Risk Subdomains to its 7 parent domains, so we can report
agreement at both the fine (subdomain) and coarse (domain) level. Verified
against https://airisk.mit.edu/risks .

Lookup is by normalized string (lowercased, whitespace-collapsed) so minor
wording/spacing differences in the export still match.
"""

from typing import Optional

from .common import norm

# domain -> list of its subdomain labels (exact AIID/MIT wording)
_DOMAINS = {
    "Discrimination & toxicity": [
        "Unfair discrimination and misrepresentation",
        "Exposure to toxic content",
        "Unequal performance across groups",
    ],
    "Privacy & security": [
        "Compromise of privacy by obtaining, leaking or correctly inferring sensitive information",
        "AI system security vulnerabilities and attacks",
    ],
    "Misinformation": [
        "False or misleading information",
        "Pollution of information ecosystem and loss of consensus reality",
    ],
    "Malicious actors & misuse": [
        "Disinformation, surveillance, and influence at scale",
        "Cyberattacks, weapon development or use, and mass harm",
        "Fraud, scams, and targeted manipulation",
    ],
    "Human-computer interaction": [
        "Overreliance and unsafe use",
        "Loss of human agency and autonomy",
    ],
    "Socioeconomic & environmental": [
        "Power centralization and unfair distribution of benefits",
        "Increased inequality and decline in employment quality",
        "Economic and cultural devaluation of human effort",
        "Competitive dynamics",
        "Governance failure",
        "Environmental harm",
    ],
    "AI system safety, failures, & limitations": [
        "AI pursuing its own goals in conflict with human goals or values",
        "AI possessing dangerous capabilities",
        "Lack of capability or robustness",
        "Lack of transparency or interpretability",
        "AI welfare and rights",
        "Multi-agent risks",
    ],
}

_SUB_TO_DOMAIN = {norm(sub): dom for dom, subs in _DOMAINS.items() for sub in subs}


def domain_of(subdomain: str) -> Optional[str] :
    """Return the parent domain for a subdomain label, or None if unknown."""
    return _SUB_TO_DOMAIN.get(norm(subdomain))