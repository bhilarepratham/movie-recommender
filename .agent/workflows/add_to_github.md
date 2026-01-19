---
description: How to push this project to GitHub
---

Since `git` is not currently installed on your Windows system (or not in your PATH), follow these steps:

## 1. Install Git
1. Download Git for Windows: [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Run the installer (accept default settings).
3. **Important**: Restart your terminal (or VS Code) after installation.

## 2. Create a Repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. Name your repository (e.g., `movie-recommender`).
3. Click **Create repository**.

## 3. Push your code
Open your terminal in the project folder (`c:\Users\PRATHAM\Desktop\my-recommender\movie-recommender`) and run:

```powershell
# Initialize git
git init

# Add all files (I have already created a .gitignore for you)
git add .

# Commit your changes
git commit -m "Initial commit: Movie Recommender App"

# Link to your GitHub repo (replace URL with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/movie-recommender.git

# Push to GitHub
git branch -M main
git push -u origin main
```
