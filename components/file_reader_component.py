"""
FileReader Component for Langflow
==================================
Allows the agent to read files from a specified project directory.
"""

from pathlib import Path

from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, StrInput, BoolInput
from lfx.schema.message import Message
from lfx.template.field.base import Output


class FileReaderComponent(Component):
    display_name = "File Reader"
    description = "Read contents of a file from the project directory. Safe for agent tool use."
    icon = "file-text"
    name = "FileReader"

    inputs = [
        StrInput(
            name="base_directory",
            display_name="Base Directory",
            info="Root directory for file operations. Files outside this directory cannot be read.",
            value="E:\\Projects\\Council Of AIs",
        ),
        MessageTextInput(
            name="file_path",
            display_name="File Path",
            info="Path to the file to read (relative to base directory or absolute within base)",
            tool_mode=True,
        ),
        BoolInput(
            name="include_line_numbers",
            display_name="Include Line Numbers",
            info="If true, prepend line numbers to each line",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="File Contents", name="contents", method="read_file"),
    ]

    def read_file(self) -> Message:
        """Read the contents of a file safely."""
        base_dir = Path(self.base_directory).resolve()
        
        # Handle both relative and absolute paths
        requested_path = Path(self.file_path)
        if requested_path.is_absolute():
            file_path = requested_path.resolve()
        else:
            file_path = (base_dir / requested_path).resolve()
        
        # Security check: ensure file is within base directory
        try:
            file_path.relative_to(base_dir)
        except ValueError:
            error_msg = f"❌ Access denied: File '{self.file_path}' is outside the allowed directory."
            self.log(error_msg)
            return Message(text=error_msg)
        
        # Check if file exists
        if not file_path.exists():
            error_msg = f"❌ File not found: '{self.file_path}'"
            self.log(error_msg)
            return Message(text=error_msg)
        
        if not file_path.is_file():
            error_msg = f"❌ Not a file: '{self.file_path}'"
            self.log(error_msg)
            return Message(text=error_msg)
        
        # Read file contents
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            # Optionally add line numbers
            if self.include_line_numbers:
                lines = content.split("\n")
                numbered_lines = [f"{i+1}: {line}" for i, line in enumerate(lines)]
                content = "\n".join(numbered_lines)
            
            # Truncate very large files
            max_chars = 50000
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n... [Truncated, file has {len(content)} characters]"
            
            self.log(f"✅ Read {len(content)} characters from {file_path.name}")
            
            # Return with metadata
            result = f"📄 **File: {file_path.name}**\n```\n{content}\n```"
            return Message(text=result)
            
        except Exception as e:
            error_msg = f"❌ Error reading file: {e}"
            self.log(error_msg)
            return Message(text=error_msg)
