---
title: Setting Up WSL Terminal Windows
date: 2023-10-21
author: Sagar Desai
categories: [dev, tool]
tags: [linux, vs code, zsh, bash]
---

{{< katex >}}

If you're a developer working on a Windows machine and want to create a robust and efficient development environment, this blog will guide you through the process of setting up Windows Subsystem for Linux (WSL) with Zsh and Oh-My-Zsh. This environment is perfect for Python development and can be seamlessly integrated with both Visual Studio Code and PyCharm.

#### Prerequisites:

Before getting started, make sure you have the following prerequisites:

1. A Windows 10 or later machine.
2. Windows Subsystem for Linux (WSL) installed. If you don't have it installed, follow [Microsoft's official guide](https://docs.microsoft.com/en-us/windows/wsl/install) to set it up.

#### Step 1: Open the Linux Terminal

Begin by opening the Windows Subsystem for Linux (WSL) terminal. You can do this by searching for "WSL" in the Windows Start menu.

#### Step 2: Install Oh-My-Zsh

Visit the [Oh-My-Zsh website](https://ohmyz.sh/) and execute the following command in your WSL terminal to install Oh-My-Zsh:

```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

This command installs Oh-My-Zsh, a powerful Zsh configuration framework.

#### Step 3: Install Zsh (if necessary)

If you encounter an error during the Oh-My-Zsh installation, you might need to install Zsh. Run the following command in your WSL terminal:

```shell
sudo apt-get install zsh
```

Zsh is a highly customizable shell similar to Bash, and Oh-My-Zsh is built on top of it.

#### Step 4: Open WSL in Visual Studio Code or PyCharm

You can use either Visual Studio Code or PyCharm for your development work. Open your preferred IDE and ensure it's set up to work with WSL.

#### Step 5: List Hidden Files

To view hidden files in your home directory, run the following command in the WSL terminal:

```shell
ls -a
```

#### Step 6: Edit .zshrc

The `.zshrc` file is the configuration file for Zsh, much like `.bashrc` for Bash. Open it using your preferred text editor. In Visual Studio Code, you can run:

```shell
code ~/.zshrc
```

Modify the `ZSH_THEME` variable to set your preferred theme. For example:

```shell
ZSH_THEME="bira"
```

This changes the appearance of your WSL terminal.

#### Step 7: Enable Plugins

Enhance your terminal by enabling various plugins in the `.zshrc` file. For example:

```shell
# Add wisely, as too many plugins can slow down shell startup.
plugins=(git web-search python pyenv virtualenv pip)
source $ZSH/oh-my-zsh.sh
```

After making changes to `.zshrc`, save the file and exit the editor.

#### Step 8: Install Zsh-Autosuggestions (Auto-Completion)

To install the Zsh-Autosuggestions plugin for auto-completion, refer to the [GitHub repository](https://github.com/zsh-users/zsh-autosuggestions) for installation instructions. Install the plugin using these commands:

```shell
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

Add `zsh-autosuggestions` to the list of activated plugins in your `.zshrc`:

```shell
plugins=(git web-search python pyenv virtualenv pip zsh-autosuggestions)
```

Save your changes and exit the editor.

#### Step 9: Install Zsh-Syntax-Highlighting

To install the Zsh-Syntax-Highlighting plugin for syntax highlighting, refer to the [GitHub repository](https://github.com/zsh-users/zsh-syntax-highlighting/blob/master/INSTALL.md) for installation instructions. Install the plugin using these commands:

```shell
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

Add `zsh-syntax-highlighting` to the list of activated plugins in your `.zshrc`:

```shell
plugins=(git web-search python pyenv virtualenv pip zsh-autosuggestions zsh-syntax-highlighting)
```

Save your changes and exit the editor.

#### Step 10: Restart the Zsh Terminal

To apply the changes you've made, restart the Zsh terminal. Simply type:

```shell
zsh
```

This will reload the Zsh configuration and apply the new theme and plugins.

You have now set up a powerful and customized development environment in WSL with Zsh and Oh-My-Zsh. This environment is ideal for Python development and can be integrated seamlessly with your preferred IDE, be it Visual Studio Code or PyCharm. Enjoy your enhanced command-line experience!

