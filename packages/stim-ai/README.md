# stim-ai

The shared model core for Stim: encoder, pointer head, artifact loading, and the root
`stim` command-line interface.

This distribution depends on `stim-systemone` plus torch, transformers, and peft.
Training and evaluation dependencies are opt-in extras; a serve-only installation
(`stim-serve`) does not install them.

PyTorch publishes no Intel macOS wheels. On that platform the torch and peft
requirements are skipped so the package remains importable, but model execution is
unavailable.
