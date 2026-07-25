---
title: Python Production
date: 2023-11-01
author: Sagar Desai
categories: [Python, production]
tags: [linux, vs code, zsh, bash]
---

{{< katex >}}

# 

Imagine this: You've developed a brilliant Python proof of concept. It works perfectly until you present it to a larger audience. Suddenly, your once flawless code can't keep up with the demand, and your reputation takes a hit.

Sounds familiar? Don't worry, there's a solution. 🎧 Eric Riddoch course, Taking Python to Production: A Professional Onboarding Guide, equips you with the tools to transition from coder to software engineer. I've completed this course and it's a game-changer. 

In the world of Python development, following best practices, tools, and workflows is essential for efficient and effective coding. Whether you're a beginner or an experienced developer, this guide will help you navigate through various aspects of Python development.

## Table of Contents

- [](#)
  - [Table of Contents](#table-of-contents)
  - [Understanding Semantic Versioning](#understanding-semantic-versioning)
  - [Setting Up the Linux Terminal in VS Code Using Zsh](#setting-up-the-linux-terminal-in-vs-code-using-zsh)
  - [Managing Multiple Python Versions with Pyenv](#managing-multiple-python-versions-with-pyenv)
  - [VS Code Extensions for Python Development](#vs-code-extensions-for-python-development)
  - [Version Control with Git](#version-control-with-git)
  - [Creating Virtual Environments](#creating-virtual-environments)
  - [Package Building in Python](#package-building-in-python)
  - [Software Testing](#software-testing)
  - [CI/CD](#cicd)
  - [Book to read around DevOps](#book-to-read-around-devops)
  - [Template repo to get started](#template-repo-to-get-started)

## Understanding Semantic Versioning

When working on Python projects, understanding semantic versioning is crucial. It helps in managing package versions and indicates the nature of changes in a version number.

Semantic versioning, often referred to as SemVer, follows the format `MAJOR.MINOR.PATCH`:

- **MAJOR** version when you make incompatible API changes.
- **MINOR** version when you add functionality in a backward-compatible manner.
- **PATCH** version when you make backward-compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the `MAJOR.MINOR.PATCH` format. This versioning system ensures that developers and users can quickly identify the impact of changes.

## Setting Up the Linux Terminal in VS Code Using Zsh

A powerful development environment is crucial, and setting up your Linux terminal in VS Code using Zsh can enhance your workflow. Here's how you can do it:

1. Open the Linux terminal.
2. Install Zsh by running the following command if it's not already installed:

   ```bash
   sudo apt-get install zsh
   ```

3. Install Oh My Zsh by executing the following command:

   ```bash
   sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
   ```

4. Customize your Zsh environment by editing the `.zshrc` file. You can change the ZSH_THEME to your preferred theme.

   ```bash
   ZSH_THEME="bira"
   ```

5. Explore and install various plugins to enhance your development experience. Popular plugins include zsh-autosuggestions and zsh-syntax-highlighting.

6. Reload Zsh after each change in the configuration.

## Managing Multiple Python Versions with Pyenv

There are situations where you need to work with different Python versions on the same machine. This is where Pyenv comes in handy. Here's how you can manage multiple Python versions using Pyenv:

1. Install Pyenv by running:

   ```bash
   curl https://pyenv.run | bash
   ```

2. Add the required lines to your `.zshrc` file to initialize Pyenv:

   ```bash
   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init --path)"
   ```

3. Install the necessary build dependencies for Pyenv:

   ```bash
   sudo apt-get update
   sudo apt-get install build-essential libssl-dev zlib1g-dev \
   libbz2-dev libreadline-dev libsqlite3-dev curl \
   libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
   ```

4. Pyenv allows you to switch between Python versions seamlessly, making it a versatile tool for Python development.

## VS Code Extensions for Python Development

Enhance your Python development experience in VS Code with these useful extensions:

		a. Markdown All in One
		b. Python
		c. Pylance
		d. Gitlens
		e. Pylint
		f. Black formatter
		g. Isort - sorts the imports
		h. Mypy - checks for type hintings
		i. Error Lens
		j. YAML
		k. Path intellecence 
		l. Even better toml
    m. Remote SSH
		n. WSL
		o. Night Owl - sarah.dranser
		p. Docker
		q. Kubernetes
		r. Find it fast
		s. Dev Container
		t. code runner
		u. excel viewer
		v. todo
    w. drawio
    x. polacode-2022
    y. spell checker
	2. Optional
		a. Flake8
		b. Darker - apply reformatting with some relaxing/ incremental adoption
		c. Makefile Tools
 

These extensions can significantly improve your code quality and development productivity.

## Version Control with Git

Version control is fundamental for collaboration and code management. Git is a widely used version control system that allows you to track changes, collaborate with others, and maintain a history of your codebase. You can use services like GitHub or GitLab to host your repositories.

## Creating Virtual Environments

Virtual environments are essential for isolating Python dependencies and managing project-specific packages. You can create virtual environments using tools like `virtualenv` or `venv`. This practice ensures that your project dependencies do not interfere with system-wide packages.

## Package Building in Python

Packaging your Python code for distribution is a key part of software development. Understanding how to create packages, manage dependencies, and use tools like `setuptools` is crucial for sharing your work with others.

Here are the corrected and cleaner instructions for managing Python packages:

**Managing Python Packages**

1. The simplest way to manage imports is to append the system path.

2. An alternative approach is to use the `setuptools` library to create a distribution package and install it, granting access to files and nested variables. You can do this by:
   a. Building and installing the package on the go with `pip install ./`
   b. Making your current directory a package and referring to it with `pip install --editable ./`

3. The source distribution method for package installation, however, has several disadvantages, including:
   a. Making assumptions about the customer's machine, such as requiring "gcc" for running "gcc numpy/*.c"
   b. Slowness, as packages like numpy, which rely on C/C++, need code compilation before execution
   c. Insecurity, as the setup is executed in the background from the main setup file
   d. The source distribution can be created with the command `python setup.py build sdist`

4. The next method involves using the `.whl` (wheel) format. To use this method:
   a. Install the wheel package with `pip install wheel`
   b. Build a wheel distribution with `pip setup.py bdist_wheel`
   c. [Learn more about Python wheels](https://realpython.com/python-wheels/#:~:text=and%20its%20dependencies.-,What%20Is%20a%20Python%20Wheel%3F,a%20type%20of%20built%20distribution.)

5. Different from package dependencies, build dependencies can be added using `setup_requires`. However, there are some disadvantages to this method:
   a. For complex libraries, external packages might be required, leading to additional dependencies
   b. Remember that these dependencies are resolved before your environment file is created.

6. The final package, the "build," is defined in a `pyproject.toml` file, and dependencies can be added there.

7. To include non-Python files in the package, you can use a `MANIFEST.in` file. Be cautious about the file paths, as caching can cause issues. Refer to the [Python documentation](https://docs.python.org/3/distutils/commandref.html#sdist-cmd) for details.

8. A more robust approach is to specify non-Python files in the `pyproject.toml` file under `[tool.setuptools.package-data]`.

9. Instead of using a `requirements.txt` file, it's recommended to list dependencies in the `pyproject.toml` file.

10. Poetry is another tool worth exploring for package management.

11. Optional dependencies can help reduce the footprint of packages. To include optional dependencies:
   a. Add optional dependencies in the `pyproject.toml` file under `project.optional-dependencies`.
   b. Install the optional dependencies with `pip install project_name[optional_dep]`.

12. To assess the health of Python packages, you can use the Snyk advisor website: [Snyk Advisor](https://snyk.io/advisor/python). Check the contributors, security issues, and other factors before adding any dependency to your project.

**Reproducibility**

1. To ensure reproducibility, document your libraries and avoid using `pip install package_name` directly. Instead, add the library to either `requirements.txt` or `pyproject.toml` and run it from there.

2. You can use `pipdeptree` to display dependencies and their dependencies within your package.

3. Document dependencies with exact pinned versions, specifying the Python version as well.

4. While the Poetry tool can be helpful, it may introduce a lot of package dependencies, making it challenging to work with.

5. Consider optional dependencies, such as linting tools, that do not need to be shipped with your package. For example, tools like Ruff and mypy can be marked as optional dependencies. Install them with `pip install ".[dev]"`.

**Document bash/sh/shell commands**
- Use Taskfile by adriancooney
- install build, dev and all to make easy to run file

---

## Software Testing
- Study software testing, test driven approach is better for sustainable package

## CI/CD
- CI/CD is important to be production ready from start
- tools GithubActions, Gitlab, Argo etc.

## Book to read around DevOps
- The Phoenix Project: A Novel about It, Devops, and Helping Your Business Win
- The Unicorn Project: A Novel about Developers, Digital Disruption, and Thriving in the Age of Data

## Template repo to get started
 - [Start template](https://github.com/SDcodehub/python-course-cookiecutter-v2/)

---
By following these tips and best practices, you can enhance your Python development workflow, write cleaner code, and manage your projects more efficiently. Stay updated with the latest Python versions, tools, and best practices to be a more effective Python developer.

For more in-depth information on the topics covered in this blog post, you can explore the provided links and resources:

- [Semantic Versioning](https://semver.org/)
- [Setting Up the Linux Terminal in VS Code Using Zsh](#setting-up-linux-terminal-in-vs-code-using-zsh)
- [Managing Multiple Python Versions with Pyenv](#managing-multiple-python-versions-with-pyenv)
- [VS Code Extensions for Python Development](#vs-code-extensions-for-python-development)
- [Version Control with Git](#version-control-with-git)
- [Creating Virtual Environments](#creating-virtual-environments)
- [Package Building in Python](#package-building-in-python)
- [Taking Python to Production: A Professional Onboarding Guide](https://www.udemy.com/course/setting-up-the-linux-terminal-for-software-development)
- [Taskfile by adriancooney](https://github.com/adriancooney/Taskfile/blob/master/Taskfile.template)

Happy coding!
