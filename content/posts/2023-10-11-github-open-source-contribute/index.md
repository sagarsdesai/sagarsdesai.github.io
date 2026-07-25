---
title: Open Source Contribution
date: 2023-10-09
author: Sagar Desai
categories: [Open-Source-Contribution]
tags: [git, gihub]
image: /path/to/featured-image.jpg
---

## Section 1: Open Source Contribution

Following are my learnings for open source contribution on github

### Subsection 1.1: How to start?

Contributing to open-source projects with Git involves a series of steps and best practices. Here's a guide on how to make contributions to open-source projects using Git:

1. **Choose a Project:**
   Select an open-source project you want to contribute to. Look for projects that align with your interests and expertise. You can find these projects on platforms like GitHub, GitLab, or Bitbucket.

2. **Fork the Repository:**
   On the project's hosting platform (e.g., GitHub), fork the repository by clicking the "Fork" button. This creates a copy of the project in your account.

3. **Clone Your Fork:**
   After forking, clone your forked repository to your local machine. Use the following command:

   ```bash
   git clone <your-fork-repository-url>
   ```

4. **Set Up a Remote:**
   To keep your forked repository in sync with the original project, add a remote reference to the original repository:

   ```bash
   git remote add upstream <original-repository-url>
   ```

5. **Create a New Branch:**
   Create a new branch for your contribution. It's a good practice to give your branch a descriptive name related to the feature or issue you're working on:

   ```bash
   git checkout -b feature/your-feature-name
   ```

6. **Make Changes:**
   Make your code changes or fix bugs in the new branch.

7. **Commit Changes:**
   Commit your changes with informative commit messages:

   ```bash
   git add .
   git commit -m "Your descriptive commit message here"
   ```

8. **Push Changes:**
   Push your changes to your forked repository:

   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request (PR):**
   Go to your forked repository on the platform and click "New Pull Request." Make sure to select the original repository as the base branch and your fork as the compare branch. Provide a detailed description of your changes in the PR.

10. **Code Review:**
    The project maintainers or other contributors will review your PR. Be ready to address their feedback and make necessary changes.

11. **Merge Changes:**
    Once your PR is approved, it will be merged into the original repository.

12. **Keep Your Fork Updated:**
    Regularly fetch changes from the original repository to keep your fork up to date. This helps avoid conflicts:

    ```bash
    git fetch upstream
    git checkout master
    git merge upstream/master
    git push origin master
    ```

13. **Stay Engaged:**
    Continue contributing, fixing issues, and adding features. Engage with the community and follow project guidelines.

Remember, each open-source project may have its own contribution guidelines and coding standards. Be sure to read and follow these guidelines closely. Additionally, be respectful and collaborative when interacting with the project maintainers and the community. Your contributions can have a significant impact on open-source projects and the broader developer community.


## Section 2: I have forked repo from main, how do I pull all the changes from original repo

To pull all the changes from the original repository (the repository you forked from) into your forked repository, you can follow these steps:

1. First, ensure you have added the original repository as a remote, typically named "upstream," as mentioned in the initial guide:

   ```bash
   git remote add upstream <original-repository-url>
   ```

2. Fetch the changes from the original repository:

   ```bash
   git fetch upstream
   ```

3. Merge the changes from the original repository into your local branch:

   If you want to update the `master` branch in your fork with the changes from the original repository's `master` branch:

   ```bash
   git checkout master
   git merge upstream/master
   ```

   If you are working on a different branch, replace `master` with the name of your branch in both the `git checkout` and `git merge` commands.

4. Push the updated branch to your fork on the remote repository (e.g., GitHub):

   ```bash
   git push origin master
   ```

   Again, replace `master` with your branch name if you're not updating the `master` branch.

By following these steps, you will sync your forked repository with the latest changes from the original repository. This is important to keep your fork up to date and to reduce the likelihood of conflicts when you contribute to the project.


## Section 3: I have made changes how do I push it to original repo

If you've made changes in the main branch of your forked repository and now want to move these changes to a new branch (e.g., `execute-code-use-docker-199`) and push it to the original repository, you can follow these steps:

1. First, make sure you've committed all your changes in the `main` branch. If you haven't committed them, do so using:

   ```bash
   git add .
   git commit -m "Your commit message"
   ```

2. Create a new branch (`execute-code-use-docker-199`) based on your current `main` branch:

   ```bash
   git checkout -b execute-code-use-docker-199
   ```

3. Now, you have the changes in your new branch. If you want to make additional changes in this branch, do so.

4. Push the new branch to your forked repository:

   ```bash
   git push origin execute-code-use-docker-199
   ```

5. To push this new branch to the original repository, you need to create a pull request. Go to your forked repository on the hosting platform (e.g., GitHub) and click on the "New Pull Request" button.

6. In the pull request, make sure to select the original repository as the base repository and branch (where you want to merge your changes), and select your forked repository as the head repository and branch (your `execute-code-use-docker-199` branch).

7. Provide a descriptive title and message for your pull request, then submit it.

8. The maintainers of the original repository will review your pull request. If it's accepted, your changes will be merged into the original repository.

Remember that you need permission to create pull requests in the original repository. If you don't have that permission, you can ask the maintainers to consider your changes by providing them with a patch or other means.