@echo off
cd /d "%~dp0"
echo === Pushing DOOM agent code to GitHub ===
git config user.name >nul 2>&1 || git config user.name "Bhanu780"
git config user.email >nul 2>&1 || git config user.email "Bhanu780@users.noreply.github.com"
git add -A -- . ":(exclude)push_to_github.bat"
git commit -m "Add memory-map agent, recurrent PPO trainer and benchmark"
git push
echo.
echo === Done. Check the messages above for errors. ===
pause
