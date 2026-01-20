

- qwen3-chat-interface.py: This file loads and uses the Qwen3 0.6B model in thinking mode.

- qwen3-chat-interface-multiturn.py: The same as above, but configured to remember the message history.
(Open and inspect these files to learn more.)

Run one of the following commands from the terminal to start the UI server:

```shell
chainlit run qwen3-chat-interface.py
```

or, if you are using uv:

```shell
uv run chainlit run qwen3-chat-interface.py
```
Running one of the commands above should open a new browser tab where you can interact with the model. If the browser tab does not open automatically, inspect the terminal command and copy the local address into your browser address bar (usually, the address is http://localhost:8000).