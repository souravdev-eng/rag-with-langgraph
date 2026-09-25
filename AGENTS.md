# Agent Guidance

These instructions apply to the entire repository.

## Project mission

Build a self-contained, progressive learning curriculum that teaches retrieval-augmented generation from foundations through advanced, agentic, autonomous, multi-agent, memory, and caching patterns. Every lesson must be understandable to a learner, executable from a clean environment, reproducible, and technically accurate.

The current priority is to review and improve every course notebook so its code teaches as clearly as its prose. Apply the notebook code-clarity rules below to existing notebooks and all new notebook work.

## Sources of truth

Before changing curriculum content:

1. Read `README.md` for the learning path, repository structure, and project-level expectations.
2. Read `docs/RAG_LEARNING_TRACKER.md` for current progress, remaining work, and the next priority.
3. Inspect the target notebook and its neighboring lessons so changes preserve the intended progression and avoid unnecessary duplication.

Keep live progress in `docs/RAG_LEARNING_TRACKER.md`; update it when notebook status changes. Keep this file focused on stable goals and working rules.

## Notebook code clarity

Write every notebook so a learner can understand the code without having to infer its purpose or data flow.

- Add concise comments inside every non-trivial code cell. Explain intent, important state changes, and how the result is used rather than merely restating Python syntax.
- Give every custom function, class, and method a short docstring that explains its purpose, inputs, return value, and any important behavior. Comment non-obvious steps within the implementation.
- Introduce each important object where it is created. State what it represents, why its configuration matters, and where it participates in the RAG pipeline.
- Explain loops, conditionals, comprehensions, and retry or iteration logic. Identify what is being traversed, what changes on each pass, and the stopping or filtering condition.
- Make transformations explicit. For chunks, embeddings, indexes, retrieval results, prompts, messages, and agent state, describe the relevant input and output types or shapes.
- Use a short Markdown introduction before a substantial example, then keep implementation-specific explanations next to the corresponding code as comments or docstrings.
- Prefer focused comments that clarify reasoning and data flow. Avoid noisy comments that only translate an obvious line into English.

## Notebook completion gate

Before declaring a notebook complete:

1. Review every code cell for functions, objects, loops, branches, and transformations that a learner may not immediately understand.
2. Add the missing explanation at the point where that code appears.
3. Execute the notebook from a clean kernel and confirm that every cell completes in order with current outputs and no errors.

A notebook is complete only when both its code and its explanation are clear, accurate, and reproducible.
