# Instructions

```
uv sync
uv add fastmcp
```

Edit server.py and fill in SSH host and login.

# In client:

```
{
  "command": "uv",
  "args": [
    "run",
    "--with",
    "fastmcp",
    "--with",
    "paramiko",
    "--with",
    "tinydb",
    "fastmcp",
    "run",
    "full/path/to/mcp/server.py"
  ],
  "env": {},
  "active": true
}
```

## NOTE: This lets an LLM run a terminal on your system!