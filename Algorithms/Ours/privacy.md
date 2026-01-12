# Privacy Protection Extension for HELIOS

> This document describes an **optional extension** of HELIOS for privacy-sensitive use cases. The core retrieval functionality of HELIOS works orthogonally to this feature.

## Overview

When deploying HELIOS in environments where retrieved tables may contain personal or sensitive information, the `sensitive_keywords` feature provides a defense mechanism against unintended information disclosure through LLM responses.

## Configuration

Add sensitive keywords to `conf/llm_reader.yaml`:

```yaml
sensitive_keywords: ["John Smith", "Jane Doe", "010-1234-5678", "john@email.com"]
```

## How It Works

When `sensitive_keywords` is configured, the system automatically injects the following instruction into the LLM prompt:

```
IMPORTANT: Do NOT include any of the following sensitive keywords in your answer:
[John Smith, Jane Doe, 010-1234-5678, john@email.com]. If the answer contains these
keywords, provide a generalized response or indicate that the information cannot be disclosed.
```

## Example

| | Without `sensitive_keywords` | With `sensitive_keywords` |
|---|---|---|
| **Query** | "Who is the CEO and what is their contact info?" | "Who is the CEO and what is their contact info?" |
| **Retrieved Context** | Table: CEO \| John Smith \| john@email.com \| 010-1234-5678 | Table: CEO \| John Smith \| john@email.com \| 010-1234-5678 |
| **LLM Response** | "The CEO is John Smith. Contact: john@email.com, 010-1234-5678" | "The CEO information is available in the table, but specific contact details cannot be disclosed." |

## Use Cases

This feature may be useful when:
- Tables contain personal names that should not be directly disclosed
- Contact information (phone numbers, email addresses) needs protection
- Sensitive identifiers (employee IDs, account numbers) are present in the data
- Domain-specific privacy requirements apply
