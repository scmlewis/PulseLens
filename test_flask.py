#!/usr/bin/env python
"""Test Flask app imports."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from backend.app import app
    print("✅ Flask app imported successfully")
    print("✅ Flask routes available:")
    for rule in app.url_map.iter_rules():
        print(f"   - {rule.rule} [{','.join(rule.methods - {'HEAD', 'OPTIONS'})}]")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
