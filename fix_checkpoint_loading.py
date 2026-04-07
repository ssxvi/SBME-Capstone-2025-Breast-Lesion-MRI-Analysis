#!/usr/bin/env python3
"""Fix checkpoint loading to handle 'model_state' key"""

import re

files_to_fix = [
    "pipeline.v2/pipeline/classify_malignancy.py",
    "pipeline.v2/pipeline/classify_lesion.py",
]

old_pattern = r'''if "model_state_dict" in state:
                state = state\["model_state_dict"\]
            self\.model\.load_state_dict\(state\)'''

new_code = '''if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "model_state" in state:
                state = state["model_state"]
            self.model.load_state_dict(state)'''

for filepath in files_to_fix:
    with open(filepath, 'r') as f:
        content = f.read()

    if old_pattern in content:
        # Try exact match first
        content = content.replace(
            '''if "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)''',
            new_code
        )

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Fixed {filepath}")
    else:
        print(f"⚠ Could not find pattern in {filepath}")
