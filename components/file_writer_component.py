"""
FileWriter Component for Langflow
==================================
Allows the agent to write/modify files with user confirmation.
"""

from pathlib import Path
from datetime import datetime

from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, StrInput, DropdownInput, BoolInput
from lfx.schema.message import Message
from lfx.template.field.base import Output


class FileWriterComponent(Component):
    display_name = "File Writer"
    description = "Write or modify files in the project directory. Creates backups before modifying."
    icon = "file-edit"
    name = "FileWriter"

    inputs = [
        StrInput(
            name="base_directory",
            display_name="Base Directory",
            info="Root directory for file operations. Files outside this directory cannot be written.",
            value="E:\\Projects\\Council Of AIs",
        ),
        MessageTextInput(
            name="file_path",
            display_name="File Path",
            info="Path to the file to write (relative to base directory)",
            tool_mode=True,
        ),
        MessageTextInput(
            name="content",
            display_name="Content",
            info="Content to write to the file",
            tool_mode=True,
        ),
        DropdownInput(
            name="write_mode",
            display_name="Write Mode",
            options=["create", "overwrite", "append"],
            value="create",
            info="create: Only if file doesn't exist. overwrite: Replace entire file. append: Add to end.",
        ),
        BoolInput(
            name="create_backup",
            display_name="Create Backup",
            info="If true, create a backup before overwriting existing files",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="dry_run",
            display_name="Dry Run (Preview Only)",
            info="If true, only show what would be written without actually writing",
            value=True,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="write_file"),
    ]

    def write_file(self) -> Message:
        """Write content to a file safely."""
        base_dir = Path(self.base_directory).resolve()
        
        # Handle path
        requested_path = Path(self.file_path)
        if requested_path.is_absolute():
            file_path = requested_path.resolve()
        else:
            file_path = (base_dir / requested_path).resolve()
        
        # Security check
        try:
            file_path.relative_to(base_dir)
        except ValueError:
            error_msg = f"❌ Access denied: Path '{self.file_path}' is outside the allowed directory."
            self.log(error_msg)
            return Message(text=error_msg)
        
        # Check write mode constraints
        file_exists = file_path.exists()
        
        if self.write_mode == "create" and file_exists:
            error_msg = f"❌ File already exists: '{file_path.name}'. Use 'overwrite' mode to replace."
            self.log(error_msg)
            return Message(text=error_msg)
        
        # Dry run mode - just show preview
        if self.dry_run:
            preview = f"""📝 **Dry Run Preview - No changes made**

**File:** `{file_path.name}`
**Full Path:** `{file_path}`
**Mode:** {self.write_mode}
**File Exists:** {file_exists}
**Content Length:** {len(self.content)} characters

**Content Preview:**
```
{self.content[:500]}{'...' if len(self.content) > 500 else ''}
```

⚠️ Set "Dry Run" to False and run again to actually write the file.
"""
            self.log(f"Dry run for {file_path.name}")
            return Message(text=preview)
        
        # Actual write operation
        try:
            # Create backup if needed
            if file_exists and self.create_backup and self.write_mode == "overwrite":
                backup_path = file_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                with open(file_path, "r", encoding="utf-8") as f:
                    original = f.read()
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(original)
                self.log(f"Created backup at {backup_path.name}")
            
            # Ensure parent directories exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            mode = "a" if self.write_mode == "append" else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(self.content)
            
            result = f"""✅ **File Written Successfully**

**File:** `{file_path.name}`
**Mode:** {self.write_mode}
**Characters Written:** {len(self.content)}
"""
            self.log(f"✅ Wrote {len(self.content)} chars to {file_path.name}")
            return Message(text=result)
            
        except Exception as e:
            error_msg = f"❌ Error writing file: {e}"
            self.log(error_msg)
            return Message(text=error_msg)
