"""Start the Guichet web app.

    python app.py                     # http://127.0.0.1:7860 (needs ANTHROPIC_API_KEY in .env)
    GUICHET_DEMO=1 python app.py      # canned answers, no key: for working on the page

See server.py for the API and settings.
"""

import os

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("server:app", host=host, port=int(os.environ.get("PORT", "7860")))
