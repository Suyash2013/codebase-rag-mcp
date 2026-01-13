# AI Coding Agent Instructions

## Project Overview

**LangflowWorkFiles** is a Langflow agent configuration utility focused on JSON processing and validation. The project handles Langflow DAG (Directed Acyclic Graph) definitions exported as JSON, with emphasis on fixing malformed JSON structures, particularly escaped quotes within nested object properties.

## Core Architecture

### Data Structure: Langflow Agent Graph Format

The primary artifact is `Simple Agent.json` — a 4400+ line Langflow configuration file defining an agentic workflow. Key structural elements:

- **edges**: Connection definitions between nodes with complex handle objects
  - Fields: `sourceHandle` and `targetHandle` contain nested JSON as string values
  - Pattern: `"sourceHandle": "{\"dataType\":\"...\",\"id\":\"...\"}"`
  - Edge `id` encodes source/target node IDs with serialized handle data

- **nodes**: Component instances (ChatInput, Chroma, ParserComponent, Prompt Template, EmbeddingModel, etc.)
  - Each node has metadata: dataType, id, position, data configuration

- **data flow**: Embeddings → Chroma (vector DB) → ParserComponent → Prompt Template → LLM chain

### Malformed JSON Problem

The core issue: Langflow exports generate edge IDs containing unescaped JSON objects, corrupting the file structure:
```json
"id": "xy-edge__EmbeddingModel-ObLRu{\"dataType\":\"EmbeddingModel\",...}-Chroma-xIUHR{...}"
```
The quotes in nested objects aren't properly escaped, breaking JSON parsers.

## Processing Utilities

### `fix_json.py` (v1)
Simple regex-based approach:
- Pattern: `"key": "{...}"` or `"key": "[...]"`
- Escapes all inner quotes: `"` → `\"`
- Limitation: Doesn't handle multi-line values or complex nesting

### `fix_json_v2.py` (v2)  
Line-by-line character parsing:
- Iterates through each line, finding `": "` delimiters
- Manually scans quoted string boundaries
- Escapes unescaped quotes: detects `"` followed by `,`, `}`, `\n`, `\r`
- Limitation: Still line-oriented; misses quotes if delimiter spans lines

**Workflow**: Try v1 first; if JSON remains malformed, use v2

## Project Conventions

1. **File Encoding**: Always use UTF-8 (specified in both scripts)
2. **JSON Escaping Logic**: Both scripts treat unescaped `"` as errors when:
   - Preceded by `": "` (key-value delimiter)
   - Not followed by a JSON delimiter (`,`, `}`, newline)
3. **Testing**: Validation via `json.tool` in `.claude/settings.local.json` (Bash environment)

## Critical Integration Points

- **Langflow Export Format**: `Simple Agent.json` is output from Langflow UI; regeneration will reproduce the malformed state
- **Claude Permissions**: `.claude/settings.local.json` allows JSON validation (`json.tool`) and git operations
- **No External Dependencies**: Scripts use only Python standard library (re, built-ins)

## Development Workflow

1. Export/acquire malformed Langflow JSON
2. Run `fix_json.py` → validate with `python -m json.tool Simple Agent.json`
3. If validation fails, run `fix_json_v2.py` → re-validate
4. Commit fixed JSON; document original issue source
5. Track `.temp` backup files created during processing

## Key Files Reference

- `Simple Agent.json` — primary Langflow workflow definition (malformed, work-in-progress)
- `fix_json.py` — regex-based JSON repair (first-pass strategy)
- `fix_json_v2.py` — line-wise JSON repair (fallback strategy)
- `.claude/settings.local.json` — Claude environment configuration for JSON tooling
- `.git/` — version control; document fixes as commits with issue context

## Notes for AI Agents

- **When modifying JSON edge structures**: Ensure all nested object quotes are escaped
- **When adding new fixes**: Consider both single-line and multi-line JSON patterns
- **When debugging**: Always validate output with `python -m json.tool` before treating as complete
- **Communication**: Langflow handles are data-dense; document changes with example before/after snippets
