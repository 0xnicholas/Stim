# stim-serve

The service layer for Stim: the `DecisionEngine` implementation, the FastAPI
application, and the `stim-serve` command-line entry point.

This distribution depends on `stim-systemone`, `stim-ai`, FastAPI, and uvicorn. Installing
it is the serve-only installation; the training and evaluation extras of `stim-ai` are
not pulled in.
