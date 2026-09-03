"""Make the repo-root scripts runnable without activating the virtualenv.

Importing this module puts the repository root (so ``openrouter_analytics`` resolves) and the
local ``.venv`` site-packages on ``sys.path``. Symlinks to the scripts are followed, so a
symlink placed anywhere on ``$PATH`` still finds the repo.
"""

import glob
import os
import sys

REPO_DIR = os.path.dirname(os.path.realpath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

for site in glob.glob(os.path.join(REPO_DIR, ".venv", "lib", "python*", "site-packages")):
    if site not in sys.path:
        sys.path.insert(0, site)
    break
