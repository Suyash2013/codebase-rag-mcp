# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LangflowWorkFiles** manages Langflow agent configurations exported as JSON. The primary focus is fixing malformed JSON structures that Langflow exports, particularly handling escaped quotes within nested object properties in edge definitions.

## Core File

**`Simple Agent.json`** (4400+ lines) - Langflow DAG workflow definition containing:
- **edges**: Node connections with complex nested JSON handles
  - `sourceHandle` and `targetHandle` fields contain stringified JSON objects
  - Edge `id` fields encode both node IDs and serialized handle data
  - Example: `"id": "xy-edge__EmbeddingModel-ObLRu{\"dataType\":\"EmbeddingModel\",...}"`
- **nodes**: Component instances (ChatInput, Chroma, EmbeddingModel, ParserComponent, Prompt Template)
- **data flow**: Embeddings → Chroma (vector DB) → ParserComponent → Prompt Template → LLM

## Common Issue: Corrupted Character Encoding

Langflow exports sometimes contain the corrupted character `œ` instead of proper quotation marks `"`. This causes Langflow to report "syntax error, line number none" even though standard JSON parsers may accept the file.

**Fix approach:**
1. Replace all `œ` with `"` using: `sed 's/œ/"/g'`
2. Escape nested quotes in edge IDs and handle strings
3. Validate with: `python -m json.tool "Simple Agent.json"`

## JSON Validation

Always validate JSON after modifications:
```bash
python -m json.tool "Simple Agent.json" > /dev/null 2>&1 && echo "Valid" || echo "Invalid"
```

## Critical Integration Points

- **Langflow Export Format**: Re-exporting from Langflow UI may reproduce malformed state
- **Encoding**: Always use UTF-8
- **Nested JSON Escaping**: Edge `id`, `sourceHandle`, and `targetHandle` fields contain JSON-as-strings requiring proper escape sequences (`\"`)

## When Modifying JSON

- Ensure all nested object quotes in edge structures are properly escaped
- Validate output with `python -m json.tool` before considering work complete
- Document changes with example before/after snippets for complex edge structures
