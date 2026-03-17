from lfx.base.data.utils import TEXT_FILE_TYPES, parallel_load_data, parse_text_file_to_data, retrieve_file_paths
from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, IntInput, MessageTextInput, MultiselectInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.template.field.base import Output

# All supported file types for projects (comprehensive list)
ALL_SUPPORTED_FILE_TYPES = TEXT_FILE_TYPES + [
    # Kotlin & Android
    "kt", "kts", "xml", "gradle", "properties",
    # Web/Frontend (optional - can be noisy)
    "js", "ts", "css", "scss", "json",
    # iOS/Swift
    "swift", "plist", "pbxproj", "xcconfig", "xcworkspacedata",
    # Config & Documentation
    "yml", "yaml", "toml", "md", "txt", "env", "example",
    # Scripts
    "bat", "ps1", "sh", "cjs",
    # IDE & Git
    "gitignore", "editorconfig", "prettierrc",
]

# Recommended defaults for KMP/Android projects (optimized for RAG)
DEFAULT_FILE_TYPES = [
    # Core source code
    "kt", "kts", "swift",
    # Android resources & config
    "xml", "properties", "toml",
    # Build scripts
    "gradle",
    # Documentation
    "md", "txt",
    # Config files
    "yml", "yaml", "env",
]


class DirectoryComponent(Component):
    display_name = "Directory"
    description = "Recursively load files from a directory."
    documentation: str = "https://docs.langflow.org/directory"
    icon = "folder"
    name = "Directory"

    inputs = [
        MessageTextInput(
            name="path",
            display_name="Path",
            info="Path to the directory to load files from. Defaults to current directory ('.')",
            value=".",
            tool_mode=True,
        ),
        MultiselectInput(
            name="types",
            display_name="File Types",
            info="File types to load. Preset with recommended types for KMP/Android projects.",
            options=ALL_SUPPORTED_FILE_TYPES,
            value=DEFAULT_FILE_TYPES,  # Preselected optimal file types
        ),
        IntInput(
            name="depth",
            display_name="Depth",
            info="Depth to search for files.",
            value=0,
        ),
        IntInput(
            name="max_concurrency",
            display_name="Max Concurrency",
            advanced=True,
            info="Maximum concurrency for loading files.",
            value=2,
        ),
        BoolInput(
            name="load_hidden",
            display_name="Load Hidden",
            advanced=True,
            info="If true, hidden files will be loaded.",
        ),
        BoolInput(
            name="recursive",
            display_name="Recursive",
            advanced=True,
            info="If true, the search will be recursive.",
        ),
        BoolInput(
            name="silent_errors",
            display_name="Silent Errors",
            advanced=True,
            info="If true, errors will not raise an exception.",
        ),
        BoolInput(
            name="use_multithreading",
            display_name="Use Multithreading",
            advanced=True,
            info="If true, multithreading will be used.",
        ),
    ]

    outputs = [
        Output(display_name="Loaded Files (Table)", name="dataframe", method="as_dataframe"),
        Output(display_name="Loaded Files (Data)", name="data", method="load_directory"),
    ]

    def load_directory(self) -> list[Data]:
        path = self.path
        types = self.types
        depth = self.depth
        max_concurrency = self.max_concurrency
        load_hidden = self.load_hidden
        recursive = self.recursive
        silent_errors = self.silent_errors
        use_multithreading = self.use_multithreading

        resolved_path = self.resolve_path(path)

        # If no types are specified, use the defaults
        if not types:
            types = DEFAULT_FILE_TYPES

        # Check if all specified types are valid
        invalid_types = [t for t in types if t not in ALL_SUPPORTED_FILE_TYPES]
        if invalid_types:
            msg = f"Invalid file types specified: {invalid_types}. Valid types are: {ALL_SUPPORTED_FILE_TYPES}"
            raise ValueError(msg)

        valid_types = types

        file_paths = retrieve_file_paths(
            resolved_path, load_hidden=load_hidden, recursive=recursive, depth=depth, types=valid_types
        )

        # Filter out build artifacts and node_modules
        ignore_patterns = [
            "/build/", "\\build\\",
            "/node_modules/", "\\node_modules\\",
            "/.gradle/", "\\.gradle\\",
            "/.idea/", "\\.idea\\",
            "/generated/", "\\generated\\",
            ".klib", ".bin", ".jar"  # Binary files
        ]

        filtered_paths = []
        for p in file_paths:
            if not any(pattern in p for pattern in ignore_patterns):
                filtered_paths.append(p)

        file_paths = filtered_paths
        self.log(f"Filtered {len(file_paths)} relevant files from project.")

        # Force silent_errors to True to handle JSON with comments and encoding issues
        loaded_data = []
        if use_multithreading:
            loaded_data = parallel_load_data(file_paths, silent_errors=True, max_concurrency=max_concurrency)
        else:
            loaded_data = [parse_text_file_to_data(file_path, silent_errors=True) for file_path in file_paths]

        valid_data = [x for x in loaded_data if x is not None and isinstance(x, Data)]

        # Log how many files were successfully loaded vs skipped
        skipped = len(file_paths) - len(valid_data)
        if skipped > 0:
            self.log(f"Successfully loaded {len(valid_data)} files, skipped {skipped} files due to parsing errors.")

        self.status = valid_data
        return valid_data

    def as_dataframe(self) -> DataFrame:
        return DataFrame(self.load_directory())
