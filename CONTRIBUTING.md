# Contributing to Quantum Forge

First off, thank you for considering contributing to Quantum Forge! Your help is greatly appreciated. Please take a moment to review these guidelines.

## Getting Started

- Ensure you have a [GitHub account](https://github.com/signup/free).
- Familiarize yourself with the project by reading the [README.md](README.md).
- Check the [open issues](../../issues), especially those tagged "good first issue" or "help wanted".

## Development Workflow

We follow a standard GitHub flow for contributions. All development happens on the `dev` branch. **Please do not commit directly to `main` or `ready-for-release`**.

1.  **Fork the Repository**: Click the "Fork" button at the top right of the [main repository page](https://github.com/Daksh-Shami/Quantum-Forge). This creates your own copy of the project.

2.  **Clone Your Fork**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/Quantum-Forge.git
    cd Quantum-Forge
    ```

3.  **Set Up Remote**: Add the original repository as the `upstream` remote.
    ```bash
    git remote add upstream https://github.com/Daksh-Shami/Quantum-Forge.git
    ```

4.  **Create a Branch**: Always create a new branch for your changes, starting from the latest `dev` branch. Choose a descriptive branch name (e.g., `feature/add-new-gate` or `fix/parser-bug`).
    ```bash
    # Fetch the latest changes from upstream
    git fetch upstream
    # Make sure your local dev branch is up-to-date
    git checkout dev
    git pull upstream dev
    # Create your new feature branch
    git checkout -b your-feature-branch-name
    ```

5.  **Make Changes**: Write your code, add tests, and ensure all tests pass (`cargo test`). Follow the existing code style. Linters will be added at some point to make this super easy.

6.  **Commit Changes**: Make clear, concise commit messages.
    ```bash
    git add .
    git commit -m "feat: Describe your feature or fix" 
    # Or use "fix:", "docs:", "style:", "refactor:", "test:", "chore:" prefixes
    ```

7.  **Pull Latest Changes**: Before pushing, pull the latest changes from the upstream `dev` branch to ensure your branch is current and to resolve any potential conflicts locally.
    ```bash
    git fetch upstream
    git rebase upstream/dev
    # Or use 'git merge upstream/dev' if you prefer, then resolve conflicts if any
    ```

8.  **Push Changes**: Push your branch to your fork.
    ```bash
    git push origin your-feature-branch-name
    ```

9.  **Open a Pull Request (PR)**:
    *   Go to your fork on GitHub.
    *   Click the "Compare & pull request" button for your branch.
    *   Ensure the base repository is `Daksh-Shami/Quantum-Forge` and the base branch is `dev`.
    *   Provide a clear title and description for your PR, explaining the changes and referencing any relevant issues (e.g., "Closes #123").
    *   **Crucially, add `Daksh-Shami` as a reviewer** to your Pull Request.

## Working with Issues

- Check the [Issues tab](../../issues) regularly for tasks that are "up for grabs".
- **Do not assign issues to yourself.**
- If you are interested in working on an issue, please **comment on the issue thread** expressing your interest.
- A maintainer will typically assign the issue to you if it's suitable. In some cases, an issue might require specialist knowledge, and this will usually be indicated in the issue description.

## Asking Questions

If you have any questions about contributing, the project structure, or run into problems, please don't hesitate to ask! You can reach out via email at: `qfteam@quantum-forge.org` or if you just have code related bugs, create an issue in the [Issues tab](../../issues).

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. (We may add a formal CODE_OF_CONDUCT.md later).

Thank you again for your contribution!
